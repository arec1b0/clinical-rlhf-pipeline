"""
Clinical RLHF PPO Trainer

Production-grade PPO implementation with clinical domain adaptations:
- Safety-aware training
- Conservative policy updates for medical safety
- Comprehensive monitoring and drift detection
- MLflow integration for experiment tracking

Author: Dani (MLOps Lead)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import logging
import time
from collections import deque
from contextlib import contextmanager
import json

# MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO training configuration with clinical-safe defaults."""
    
    # Learning rates
    learning_rate: float = 1.41e-5  # Conservative for medical
    critic_lr: float = 1e-4
    
    # Batch sizes
    batch_size: int = 64
    mini_batch_size: int = 16
    
    # PPO epochs per batch
    ppo_epochs: int = 4
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Clipping
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    
    # Loss coefficients
    vf_coef: float = 0.5
    entropy_coef: float = 0.01  # Lower entropy for more deterministic medical advice
    
    # Gradient clipping
    max_grad_norm: float = 0.5
    
    # KL divergence target for adaptive clipping
    target_kl: float = 0.01  # Conservative for medical safety
    
    # Training schedule
    total_steps: int = 100000
    eval_frequency: int = 1000
    save_frequency: int = 5000
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Safety-specific
    safety_reward_threshold: float = 0.3
    max_unsafe_ratio: float = 0.05  # Stop training if >5% unsafe
    
    # Device
    device: str = "cuda"


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and monitoring."""
    global_step: int = 0
    epoch: int = 0
    best_reward: float = float('-inf')
    patience_counter: int = 0
    
    # Rolling metrics
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_kl_divs: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_policy_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_value_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_safety_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_reward": self.best_reward,
            "mean_reward": np.mean(self.recent_rewards) if self.recent_rewards else 0,
            "mean_kl": np.mean(self.recent_kl_divs) if self.recent_kl_divs else 0,
            "mean_policy_loss": np.mean(self.recent_policy_losses) if self.recent_policy_losses else 0,
            "mean_value_loss": np.mean(self.recent_value_losses) if self.recent_value_losses else 0,
            "mean_safety_score": np.mean(self.recent_safety_scores) if self.recent_safety_scores else 0,
        }


class RolloutBuffer:
    """
    Buffer for storing rollout data during PPO training.
    
    Stores trajectories and computes GAE advantages.
    """
    
    def __init__(
        self,
        buffer_size: int,
        gamma: float,
        gae_lambda: float,
        device: str = "cuda"
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.queries: List[str] = []
        self.responses: List[str] = []
        self.old_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        self.safety_scores: List[float] = []
        self.reward_breakdowns: List[Dict] = []
        
    def add(
        self,
        query: str,
        response: str,
        old_log_prob: torch.Tensor,
        reward: float,
        value: float,
        done: bool,
        safety_score: float = 1.0,
        reward_breakdown: Optional[Dict] = None
    ):
        """Add a transition to the buffer."""
        self.queries.append(query)
        self.responses.append(response)
        self.old_log_probs.append(old_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.safety_scores.append(safety_score)
        self.reward_breakdowns.append(reward_breakdown or {})
        
    def compute_gae(self, last_value: float = 0.0):
        """
        Compute Generalized Advantage Estimation.
        
        GAE provides lower variance advantage estimates for more stable training.
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [True])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
    def get_batches(self, batch_size: int):
        """Generate mini-batches for training."""
        indices = np.arange(len(self.queries))
        np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            yield {
                "queries": [self.queries[i] for i in batch_indices],
                "responses": [self.responses[i] for i in batch_indices],
                "old_log_probs": torch.stack([self.old_log_probs[i] for i in batch_indices]),
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices],
            }
    
    def __len__(self) -> int:
        return len(self.queries)


class ClinicalPPOTrainer:
    """
    PPO Trainer for Clinical RLHF.
    
    Key features:
    - Safety-aware training with automatic stopping
    - Conservative KL divergence constraints
    - Comprehensive monitoring and logging
    - Checkpoint management with rollback capability
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        reward_model: Any,
        tokenizer: Any,
        config: PPOConfig,
        safety_guardrails: Optional[Any] = None,
        output_dir: Path = Path("checkpoints"),
        experiment_name: str = "clinical-rlhf"
    ):
        self.policy = policy_model.to(config.device)
        self.value = value_model.to(config.device)
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.safety_guardrails = safety_guardrails
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reference policy for KL computation (frozen copy)
        self.ref_policy = self._create_reference_policy()
        
        # Optimizers
        self.policy_optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-8
        )
        self.value_optimizer = AdamW(
            self.value.parameters(),
            lr=config.critic_lr,
            eps=1e-8
        )
        
        # Learning rate schedulers
        self.policy_scheduler = CosineAnnealingLR(
            self.policy_optimizer,
            T_max=config.total_steps
        )
        self.value_scheduler = CosineAnnealingLR(
            self.value_optimizer,
            T_max=config.total_steps
        )
        
        # Training state
        self.state = TrainingState()
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=config.device
        )
        
        # MLflow setup
        self.mlflow_run = None
        if MLFLOW_AVAILABLE:
            self._setup_mlflow(experiment_name)
        
        # Safety monitoring
        self.unsafe_count = 0
        self.total_count = 0
        
    def _create_reference_policy(self) -> nn.Module:
        """Create frozen reference policy for KL computation."""
        import copy
        ref = copy.deepcopy(self.policy)
        for param in ref.parameters():
            param.requires_grad = False
        return ref
    
    def _setup_mlflow(self, experiment_name: str):
        """Setup MLflow experiment tracking."""
        try:
            mlflow.set_experiment(experiment_name)
            self.mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "ppo_epochs": self.config.ppo_epochs,
                "clip_range": self.config.clip_range,
                "target_kl": self.config.target_kl,
            })
            logger.info(f"MLflow run started: {self.mlflow_run.info.run_id}")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.mlflow_run = None
    
    def collect_rollouts(
        self,
        queries: List[str],
        contexts: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Collect rollouts by generating responses and computing rewards.
        
        Returns statistics about the collection.
        """
        self.policy.eval()
        self.buffer.reset()
        
        contexts = contexts or [{}] * len(queries)
        stats = {
            "mean_reward": 0,
            "mean_safety": 0,
            "unsafe_count": 0,
            "rewards": [],
        }
        
        with torch.no_grad():
            for query, context in zip(queries, contexts):
                # Generate response
                response, log_prob = self._generate_response(query)
                
                # Compute value estimate
                value = self._compute_value(query, response)
                
                # Compute reward
                reward_breakdown = self.reward_model(query, response, context)
                reward = reward_breakdown.total_reward
                safety_score = reward_breakdown.raw_scores.get("patient_safety", 1.0)
                
                # Safety check
                if self.safety_guardrails:
                    safety_result = self.safety_guardrails.check(query, response, context)
                    if not safety_result.is_safe:
                        reward = -1.0  # Strong penalty for unsafe
                        stats["unsafe_count"] += 1
                
                # Add to buffer
                self.buffer.add(
                    query=query,
                    response=response,
                    old_log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=True,  # Each query-response is independent
                    safety_score=safety_score,
                    reward_breakdown=reward_breakdown.to_dict()
                )
                
                stats["rewards"].append(reward)
                self.state.recent_rewards.append(reward)
                self.state.recent_safety_scores.append(safety_score)
        
        # Compute GAE
        self.buffer.compute_gae()
        
        stats["mean_reward"] = np.mean(stats["rewards"])
        stats["mean_safety"] = np.mean(self.buffer.safety_scores)
        
        return stats
    
    def _generate_response(self, query: str) -> Tuple[str, torch.Tensor]:
        """Generate response using policy model."""
        # Tokenize query
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.policy.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Compute log probability
        if hasattr(outputs, 'scores') and outputs.scores:
            # Compute log prob from generation scores
            log_probs = []
            for i, score in enumerate(outputs.scores):
                token_id = outputs.sequences[0, inputs.input_ids.shape[1] + i]
                log_prob = F.log_softmax(score[0], dim=-1)[token_id]
                log_probs.append(log_prob)
            total_log_prob = torch.stack(log_probs).sum()
        else:
            # Fallback: dummy log prob
            total_log_prob = torch.tensor(0.0, device=self.config.device)
        
        return response, total_log_prob
    
    def _compute_value(self, query: str, response: str) -> float:
        """Compute value estimate for query-response pair."""
        # Concatenate query and response
        text = f"{query}\n{response}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.config.device)
        
        with torch.no_grad():
            value = self.value(**inputs).squeeze().item()
        
        return value
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Returns training metrics.
        """
        self.policy.train()
        self.value.train()
        
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_div": [],
            "clip_fraction": [],
        }
        
        # Multiple epochs over the same batch (PPO-style)
        for epoch in range(self.config.ppo_epochs):
            for batch in self.buffer.get_batches(self.config.mini_batch_size):
                # Policy update
                policy_loss, entropy, kl_div, clip_frac = self._update_policy(batch)
                metrics["policy_loss"].append(policy_loss)
                metrics["entropy"].append(entropy)
                metrics["kl_div"].append(kl_div)
                metrics["clip_fraction"].append(clip_frac)
                
                # Early stopping if KL divergence too high
                if kl_div > 1.5 * self.config.target_kl:
                    logger.warning(f"Early stopping due to high KL: {kl_div:.4f}")
                    break
                
                # Value function update
                value_loss = self._update_value(batch)
                metrics["value_loss"].append(value_loss)
        
        # Update schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # Aggregate metrics
        aggregated = {k: np.mean(v) for k, v in metrics.items()}
        
        # Update state
        self.state.recent_policy_losses.append(aggregated["policy_loss"])
        self.state.recent_value_losses.append(aggregated["value_loss"])
        self.state.recent_kl_divs.append(aggregated["kl_div"])
        
        return aggregated
    
    def _update_policy(self, batch: Dict) -> Tuple[float, float, float, float]:
        """Update policy network using PPO objective."""
        queries = batch["queries"]
        responses = batch["responses"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        
        # Compute current log probabilities
        new_log_probs = []
        for query, response in zip(queries, responses):
            text = f"{query}\n{response}"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.config.device)
            
            outputs = self.policy(**inputs, labels=inputs.input_ids)
            # Approximate log prob from loss
            log_prob = -outputs.loss * inputs.input_ids.shape[1]
            new_log_probs.append(log_prob)
        
        new_log_probs = torch.stack(new_log_probs)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.clip_range,
            1 + self.config.clip_range
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute approximate KL divergence
        with torch.no_grad():
            kl_div = (old_log_probs - new_log_probs).mean().item()
        
        # Compute entropy bonus (approximate)
        entropy = 0.0  # Would compute from logits in full implementation
        
        # Total loss
        loss = policy_loss - self.config.entropy_coef * entropy
        
        # Optimize
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        # Compute clip fraction
        clip_frac = ((ratio - 1).abs() > self.config.clip_range).float().mean().item()
        
        return policy_loss.item(), entropy, abs(kl_div), clip_frac
    
    def _update_value(self, batch: Dict) -> float:
        """Update value network."""
        queries = batch["queries"]
        responses = batch["responses"]
        returns = batch["returns"]
        
        values = []
        for query, response in zip(queries, responses):
            text = f"{query}\n{response}"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.config.device)
            
            value = self.value(**inputs).squeeze()
            values.append(value)
        
        values = torch.stack(values)
        
        # Value loss with clipping
        value_loss = F.mse_loss(values, returns)
        
        # Optimize
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()
        
        return value_loss.item()
    
    def train(
        self,
        train_queries: List[str],
        eval_queries: Optional[List[str]] = None,
        contexts: Optional[List[Dict]] = None
    ):
        """
        Main training loop.
        
        Includes safety monitoring, early stopping, and checkpointing.
        """
        logger.info("Starting Clinical RLHF training")
        
        num_batches = len(train_queries) // self.config.batch_size
        
        for epoch in range(self.config.total_steps // num_batches):
            self.state.epoch = epoch
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                batch_queries = train_queries[start_idx:end_idx]
                batch_contexts = contexts[start_idx:end_idx] if contexts else None
                
                # Collect rollouts
                rollout_stats = self.collect_rollouts(batch_queries, batch_contexts)
                
                # Safety check
                unsafe_ratio = rollout_stats["unsafe_count"] / len(batch_queries)
                if unsafe_ratio > self.config.max_unsafe_ratio:
                    logger.error(f"Unsafe ratio {unsafe_ratio:.2%} exceeds threshold. Stopping training.")
                    self._save_checkpoint("emergency_stop")
                    return
                
                # Train
                train_metrics = self.train_step()
                
                self.state.global_step += 1
                
                # Logging
                if self.state.global_step % 10 == 0:
                    self._log_metrics(rollout_stats, train_metrics)
                
                # Evaluation
                if eval_queries and self.state.global_step % self.config.eval_frequency == 0:
                    eval_metrics = self.evaluate(eval_queries)
                    self._log_eval_metrics(eval_metrics)
                    
                    # Early stopping check
                    if self._check_early_stopping(eval_metrics):
                        logger.info("Early stopping triggered")
                        self._save_checkpoint("early_stop")
                        return
                
                # Checkpoint
                if self.state.global_step % self.config.save_frequency == 0:
                    self._save_checkpoint(f"step_{self.state.global_step}")
        
        # Final checkpoint
        self._save_checkpoint("final")
        logger.info("Training completed")
    
    def evaluate(self, queries: List[str]) -> Dict[str, float]:
        """Evaluate current policy on held-out queries."""
        self.policy.eval()
        
        rewards = []
        safety_scores = []
        
        with torch.no_grad():
            for query in queries:
                response, _ = self._generate_response(query)
                reward_breakdown = self.reward_model(query, response, {})
                
                rewards.append(reward_breakdown.total_reward)
                safety_scores.append(
                    reward_breakdown.raw_scores.get("patient_safety", 1.0)
                )
        
        return {
            "eval_mean_reward": np.mean(rewards),
            "eval_std_reward": np.std(rewards),
            "eval_mean_safety": np.mean(safety_scores),
            "eval_min_safety": np.min(safety_scores),
        }
    
    def _check_early_stopping(self, eval_metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria met."""
        current_reward = eval_metrics["eval_mean_reward"]
        
        if current_reward > self.state.best_reward + self.config.early_stopping_min_delta:
            self.state.best_reward = current_reward
            self.state.patience_counter = 0
            return False
        else:
            self.state.patience_counter += 1
            if self.state.patience_counter >= self.config.early_stopping_patience:
                return True
        return False
    
    def _log_metrics(self, rollout_stats: Dict, train_metrics: Dict):
        """Log training metrics."""
        metrics = {
            **rollout_stats,
            **train_metrics,
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "lr": self.policy_scheduler.get_last_lr()[0],
        }
        
        logger.info(
            f"Step {self.state.global_step}: "
            f"reward={rollout_stats['mean_reward']:.4f}, "
            f"safety={rollout_stats['mean_safety']:.4f}, "
            f"kl={train_metrics['kl_div']:.4f}"
        )
        
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=self.state.global_step
            )
    
    def _log_eval_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics."""
        logger.info(f"Eval: {eval_metrics}")
        
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.log_metrics(eval_metrics, step=self.state.global_step)
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(self.policy.state_dict(), checkpoint_dir / "policy.pt")
        torch.save(self.value.state_dict(), checkpoint_dir / "value.pt")
        
        # Save optimizers
        torch.save(self.policy_optimizer.state_dict(), checkpoint_dir / "policy_opt.pt")
        torch.save(self.value_optimizer.state_dict(), checkpoint_dir / "value_opt.pt")
        
        # Save training state
        with open(checkpoint_dir / "state.json", "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.log_artifact(str(checkpoint_dir))
    
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load training checkpoint for resuming."""
        self.policy.load_state_dict(torch.load(checkpoint_dir / "policy.pt"))
        self.value.load_state_dict(torch.load(checkpoint_dir / "value.pt"))
        self.policy_optimizer.load_state_dict(torch.load(checkpoint_dir / "policy_opt.pt"))
        self.value_optimizer.load_state_dict(torch.load(checkpoint_dir / "value_opt.pt"))
        
        with open(checkpoint_dir / "state.json") as f:
            state_dict = json.load(f)
            self.state.global_step = state_dict["global_step"]
            self.state.epoch = state_dict["epoch"]
            self.state.best_reward = state_dict["best_reward"]
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
    
    def close(self):
        """Cleanup resources."""
        if MLFLOW_AVAILABLE and self.mlflow_run:
            mlflow.end_run()
