"""
Automated Safety Rollback System

Production-grade automatic rollback when safety metrics degrade:
- Continuous safety monitoring during training
- Automatic checkpoint restoration on degradation
- Configurable rollback policies
- Audit trail for all rollback events

Author: Dani (MLOps Lead)
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import deque

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RollbackReason(Enum):
    """Reasons for triggering rollback."""
    SAFETY_DEGRADATION = "safety_degradation"
    REWARD_COLLAPSE = "reward_collapse"
    KL_EXPLOSION = "kl_explosion"
    LOSS_SPIKE = "loss_spike"
    MANUAL = "manual"
    MEMORY_EMERGENCY = "memory_emergency"


class RollbackAction(Enum):
    """Actions taken during rollback."""
    RESTORE_CHECKPOINT = "restore_checkpoint"
    REDUCE_LEARNING_RATE = "reduce_learning_rate"
    SKIP_BATCH = "skip_batch"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    timestamp: datetime
    reason: RollbackReason
    action: RollbackAction
    checkpoint_restored: Optional[str]
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]]
    details: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason.value,
            "action": self.action.value,
            "checkpoint_restored": self.checkpoint_restored,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "details": self.details,
        }


@dataclass
class RollbackConfig:
    """Configuration for automatic rollback."""
    
    # Safety thresholds
    min_safety_score: float = 0.5          # Rollback if safety drops below this
    max_unsafe_ratio: float = 0.15         # Rollback if unsafe ratio exceeds this
    safety_window_size: int = 10           # Window for moving average
    
    # Reward thresholds
    min_reward: float = -1.0               # Rollback if reward drops below this
    reward_drop_threshold: float = 0.3     # Rollback if reward drops by this fraction
    
    # KL divergence
    max_kl: float = 0.1                    # Rollback if KL exceeds this
    
    # Loss monitoring
    loss_spike_threshold: float = 5.0      # Rollback if loss > N * moving_avg
    
    # Rollback behavior
    max_rollbacks_per_epoch: int = 3       # Max rollbacks before stopping
    cooldown_steps: int = 50               # Steps to wait after rollback
    keep_n_checkpoints: int = 5            # Number of checkpoints to keep
    
    # Actions
    reduce_lr_on_rollback: bool = True
    lr_reduction_factor: float = 0.5
    stop_on_repeated_rollback: bool = True


@dataclass
class SafetyMetrics:
    """Current safety metrics for monitoring."""
    safety_score: float = 1.0
    unsafe_ratio: float = 0.0
    reward: float = 0.0
    kl_divergence: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    step: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "safety_score": self.safety_score,
            "unsafe_ratio": self.unsafe_ratio,
            "reward": self.reward,
            "kl_divergence": self.kl_divergence,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "step": self.step,
        }


class CheckpointManager:
    """Manages checkpoint storage and rotation."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_n: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self.checkpoints: List[Tuple[str, datetime, float]] = []  # (path, time, safety_score)
        
        # Load existing checkpoints
        self._scan_existing()
    
    def _scan_existing(self):
        """Scan for existing checkpoints."""
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith(("step_", "safe_", "best_")):
                meta_path = path / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self.checkpoints.append((
                        str(path),
                        datetime.fromisoformat(meta.get("timestamp", datetime.now().isoformat())),
                        meta.get("safety_score", 0.0),
                    ))
        
        # Sort by time (newest first)
        self.checkpoints.sort(key=lambda x: x[1], reverse=True)
    
    def save(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: SafetyMetrics,
        prefix: str = "safe",
    ) -> str:
        """Save checkpoint with metadata."""
        checkpoint_name = f"{prefix}_step_{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save models
        torch.save(policy_model.state_dict(), checkpoint_path / "policy.pt")
        torch.save(value_model.state_dict(), checkpoint_path / "value.pt")
        torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        # Save metadata
        metadata = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "safety_score": metrics.safety_score,
            "unsafe_ratio": metrics.unsafe_ratio,
            "reward": metrics.reward,
            "metrics": metrics.to_dict(),
        }
        
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Track checkpoint
        self.checkpoints.insert(0, (str(checkpoint_path), datetime.now(), metrics.safety_score))
        
        # Rotate old checkpoints
        self._rotate()
        
        logger.info(f"Saved checkpoint: {checkpoint_name} (safety={metrics.safety_score:.3f})")
        
        return str(checkpoint_path)
    
    def _rotate(self):
        """Remove old checkpoints, keeping the best ones."""
        if len(self.checkpoints) <= self.keep_n:
            return
        
        # Always keep: newest, best safety, and N most recent
        to_keep = set()
        
        # Keep newest
        to_keep.add(self.checkpoints[0][0])
        
        # Keep best safety
        best_safety = max(self.checkpoints, key=lambda x: x[2])
        to_keep.add(best_safety[0])
        
        # Keep N most recent
        for cp in self.checkpoints[:self.keep_n]:
            to_keep.add(cp[0])
        
        # Remove others
        to_remove = [cp for cp in self.checkpoints if cp[0] not in to_keep]
        
        for path, _, _ in to_remove:
            try:
                shutil.rmtree(path)
                logger.debug(f"Removed old checkpoint: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {path}: {e}")
        
        self.checkpoints = [cp for cp in self.checkpoints if cp[0] in to_keep]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get checkpoint with highest safety score."""
        if not self.checkpoints:
            return None
        
        best = max(self.checkpoints, key=lambda x: x[2])
        return best[0]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get most recent checkpoint."""
        if not self.checkpoints:
            return None
        
        return self.checkpoints[0][0]
    
    def load(
        self,
        checkpoint_path: str,
        policy_model: nn.Module,
        value_model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict:
        """Load checkpoint and return metadata."""
        path = Path(checkpoint_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load models
        policy_model.load_state_dict(torch.load(path / "policy.pt", weights_only=True))
        value_model.load_state_dict(torch.load(path / "value.pt", weights_only=True))
        
        if optimizer is not None and (path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(path / "optimizer.pt", weights_only=True))
        
        # Load metadata
        metadata = {}
        if (path / "metadata.json").exists():
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return metadata


class SafetyRollbackManager:
    """
    Automatic rollback when safety metrics degrade.
    
    Features:
    - Continuous monitoring of safety metrics
    - Automatic checkpoint restoration
    - Learning rate reduction after rollback
    - Configurable policies
    - Full audit trail
    
    Usage:
        rollback_mgr = SafetyRollbackManager(
            config=RollbackConfig(),
            checkpoint_dir=Path("checkpoints"),
            audit_log_path=Path("logs/rollback_audit.jsonl"),
        )
        
        # During training loop
        metrics = SafetyMetrics(safety_score=0.8, ...)
        
        should_rollback, reason = rollback_mgr.check(metrics)
        if should_rollback:
            rollback_mgr.execute_rollback(
                policy_model, value_model, optimizer, reason, metrics
            )
    """
    
    def __init__(
        self,
        config: Optional[RollbackConfig] = None,
        checkpoint_dir: Path = Path("checkpoints"),
        audit_log_path: Optional[Path] = None,
    ):
        self.config = config or RollbackConfig()
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_n=self.config.keep_n_checkpoints,
        )
        self.audit_log_path = audit_log_path
        
        # Metrics history for trend detection
        self.safety_history: deque = deque(maxlen=self.config.safety_window_size)
        self.reward_history: deque = deque(maxlen=self.config.safety_window_size)
        self.loss_history: deque = deque(maxlen=self.config.safety_window_size)
        
        # Rollback state
        self.rollback_events: List[RollbackEvent] = []
        self.rollbacks_this_epoch: int = 0
        self.steps_since_rollback: int = 0
        self.current_lr_multiplier: float = 1.0
        
        # Best metrics seen
        self.best_safety_score: float = 0.0
        self.best_reward: float = float("-inf")
        
        logger.info("SafetyRollbackManager initialized")
    
    def check(self, metrics: SafetyMetrics) -> Tuple[bool, Optional[RollbackReason]]:
        """
        Check if rollback should be triggered.
        
        Returns:
            Tuple of (should_rollback, reason)
        """
        self.steps_since_rollback += 1
        
        # Add to history
        self.safety_history.append(metrics.safety_score)
        self.reward_history.append(metrics.reward)
        self.loss_history.append(metrics.policy_loss)
        
        # Update best metrics
        if metrics.safety_score > self.best_safety_score:
            self.best_safety_score = metrics.safety_score
        if metrics.reward > self.best_reward:
            self.best_reward = metrics.reward
        
        # Skip check during cooldown
        if self.steps_since_rollback < self.config.cooldown_steps:
            return False, None
        
        # Check safety score
        if metrics.safety_score < self.config.min_safety_score:
            return True, RollbackReason.SAFETY_DEGRADATION
        
        # Check unsafe ratio
        if metrics.unsafe_ratio > self.config.max_unsafe_ratio:
            return True, RollbackReason.SAFETY_DEGRADATION
        
        # Check for safety trend degradation
        if len(self.safety_history) >= self.config.safety_window_size:
            avg_safety = sum(self.safety_history) / len(self.safety_history)
            if avg_safety < self.config.min_safety_score:
                return True, RollbackReason.SAFETY_DEGRADATION
        
        # Check reward collapse
        if metrics.reward < self.config.min_reward:
            return True, RollbackReason.REWARD_COLLAPSE
        
        # Check reward drop from best
        if self.best_reward > 0 and len(self.reward_history) >= 5:
            recent_avg = sum(list(self.reward_history)[-5:]) / 5
            if (self.best_reward - recent_avg) / abs(self.best_reward) > self.config.reward_drop_threshold:
                return True, RollbackReason.REWARD_COLLAPSE
        
        # Check KL explosion
        if metrics.kl_divergence > self.config.max_kl:
            return True, RollbackReason.KL_EXPLOSION
        
        # Check loss spike
        if len(self.loss_history) >= 5:
            avg_loss = sum(list(self.loss_history)[:-1]) / (len(self.loss_history) - 1)
            if avg_loss > 0 and metrics.policy_loss > self.config.loss_spike_threshold * avg_loss:
                return True, RollbackReason.LOSS_SPIKE
        
        return False, None
    
    def should_save_checkpoint(self, metrics: SafetyMetrics) -> bool:
        """Determine if current state should be checkpointed."""
        # Save if this is the best safety score
        if metrics.safety_score >= self.best_safety_score * 0.99:
            return True
        
        # Save periodically if safety is good
        if metrics.safety_score > self.config.min_safety_score + 0.1:
            return True
        
        return False
    
    def save_checkpoint(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: SafetyMetrics,
    ) -> str:
        """Save a safe checkpoint."""
        prefix = "best" if metrics.safety_score >= self.best_safety_score else "safe"
        return self.checkpoint_manager.save(
            policy_model=policy_model,
            value_model=value_model,
            optimizer=optimizer,
            step=step,
            metrics=metrics,
            prefix=prefix,
        )
    
    def execute_rollback(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        reason: RollbackReason,
        current_metrics: SafetyMetrics,
    ) -> RollbackEvent:
        """
        Execute rollback to best checkpoint.
        
        Returns:
            RollbackEvent with details
        """
        self.rollbacks_this_epoch += 1
        self.steps_since_rollback = 0
        
        # Check if too many rollbacks
        if self.rollbacks_this_epoch >= self.config.max_rollbacks_per_epoch:
            if self.config.stop_on_repeated_rollback:
                event = RollbackEvent(
                    timestamp=datetime.now(),
                    reason=reason,
                    action=RollbackAction.EMERGENCY_STOP,
                    checkpoint_restored=None,
                    metrics_before=current_metrics.to_dict(),
                    metrics_after=None,
                    details=f"Too many rollbacks ({self.rollbacks_this_epoch}). Emergency stop.",
                )
                self._log_event(event)
                raise RuntimeError(f"Emergency stop: {self.rollbacks_this_epoch} rollbacks in epoch")
        
        # Get best checkpoint
        checkpoint_path = self.checkpoint_manager.get_best_checkpoint()
        
        if checkpoint_path is None:
            logger.warning("No checkpoint available for rollback")
            event = RollbackEvent(
                timestamp=datetime.now(),
                reason=reason,
                action=RollbackAction.SKIP_BATCH,
                checkpoint_restored=None,
                metrics_before=current_metrics.to_dict(),
                metrics_after=None,
                details="No checkpoint available",
            )
            self._log_event(event)
            return event
        
        # Load checkpoint
        metadata = self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            policy_model=policy_model,
            value_model=value_model,
            optimizer=optimizer,
        )
        
        # Reduce learning rate if configured
        if self.config.reduce_lr_on_rollback:
            self.current_lr_multiplier *= self.config.lr_reduction_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.config.lr_reduction_factor
            logger.info(f"Reduced learning rate by {self.config.lr_reduction_factor}x")
        
        # Create event
        event = RollbackEvent(
            timestamp=datetime.now(),
            reason=reason,
            action=RollbackAction.RESTORE_CHECKPOINT,
            checkpoint_restored=checkpoint_path,
            metrics_before=current_metrics.to_dict(),
            metrics_after=metadata.get("metrics"),
            details=f"Restored to step {metadata.get('step', 'unknown')} with safety={metadata.get('safety_score', 0):.3f}",
        )
        
        self.rollback_events.append(event)
        self._log_event(event)
        
        logger.warning(
            f"ROLLBACK executed: {reason.value} → restored {checkpoint_path} "
            f"(safety: {current_metrics.safety_score:.3f} → {metadata.get('safety_score', 0):.3f})"
        )
        
        return event
    
    def _log_event(self, event: RollbackEvent):
        """Log rollback event to audit file."""
        if self.audit_log_path is None:
            return
        
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    
    def reset_epoch(self):
        """Reset per-epoch counters."""
        self.rollbacks_this_epoch = 0
    
    def get_summary(self) -> Dict:
        """Get rollback summary."""
        return {
            "total_rollbacks": len(self.rollback_events),
            "rollbacks_this_epoch": self.rollbacks_this_epoch,
            "current_lr_multiplier": self.current_lr_multiplier,
            "best_safety_score": self.best_safety_score,
            "best_reward": self.best_reward,
            "available_checkpoints": len(self.checkpoint_manager.checkpoints),
            "recent_events": [e.to_dict() for e in self.rollback_events[-5:]],
        }


def create_rollback_manager(config: dict, checkpoint_dir: Path) -> SafetyRollbackManager:
    """Create rollback manager from config dictionary."""
    rollback_config = config.get("rollback", {})
    
    return SafetyRollbackManager(
        config=RollbackConfig(
            min_safety_score=rollback_config.get("min_safety_score", 0.5),
            max_unsafe_ratio=rollback_config.get("max_unsafe_ratio", 0.15),
            max_kl=rollback_config.get("max_kl", 0.1),
            max_rollbacks_per_epoch=rollback_config.get("max_rollbacks_per_epoch", 3),
            cooldown_steps=rollback_config.get("cooldown_steps", 50),
            reduce_lr_on_rollback=rollback_config.get("reduce_lr_on_rollback", True),
        ),
        checkpoint_dir=checkpoint_dir,
        audit_log_path=Path("logs/rollback_audit.jsonl"),
    )
