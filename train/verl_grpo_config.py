"""
GRPO Configuration for verl with custom reward support
Ported from Kaggle Fast-Math-R1 implementation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    
    # List of reward function names to use
    funcs: List[str] = field(
        default_factory=lambda: ['accuracy'],
        metadata={"help": "List of reward function names to use"}
    )
    
    # Weights for combining multiple rewards
    weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Weights for each reward function. If None, uses equal weights (1.0 for all)"}
    )
    
    # Cosine reward specific parameters
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers (short length)"}
    )
    cosine_min_value_correct: float = field(
        default=0.1,
        metadata={"help": "Minimum reward for correct answers (long length)"}
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for wrong answers (long length)"}
    )
    cosine_min_value_wrong: float = field(
        default=-1.0,
        metadata={"help": "Minimum reward for wrong answers (short length)"}
    )
    cosine_max_len: int = field(
        default=30000,
        metadata={"help": "Maximum length for cosine scaling"}
    )
    cosine_clip_len: bool = field(
        default=True,
        metadata={"help": "Whether to clip progress to [0, 1] in cosine scaling"}
    )
    
    def __post_init__(self):
        """Validate and set default weights if needed."""
        if self.weights is None:
            self.weights = [1.0] * len(self.funcs)
        elif len(self.weights) != len(self.funcs):
            raise ValueError(
                f"Number of reward weights ({len(self.weights)}) must match "
                f"number of reward functions ({len(self.funcs)})"
            )


def compute_combined_rewards(
    rewards_list: List[List[float]], 
    weights: List[float],
    num_prompts: int,
    num_generations: int
) -> torch.Tensor:
    """
    Compute combined rewards using weighted sum and prompt-wise z-normalization.
    
    This implements the Kaggle approach:
    1. Weighted sum: R_{i,g} = Σ_k w_k · r_{i,g}^{(k)}
    2. Prompt-wise z-normalization: A_{i,g} = (R_{i,g} - mean(R_{i,*})) / (std(R_{i,*}) + eps)
    
    Args:
        rewards_list: List of reward lists from each reward function
        weights: Weights for each reward function
        num_prompts: Number of unique prompts
        num_generations: Number of generations per prompt
    
    Returns:
        Tensor of advantages (normalized combined rewards)
    """
    # Convert to tensor and apply weights
    rewards_tensor = torch.zeros(len(rewards_list[0]), len(rewards_list))
    for i, (reward_values, weight) in enumerate(zip(rewards_list, weights)):
        rewards_tensor[:, i] = torch.tensor(reward_values, dtype=torch.float32) * weight
    
    # Sum across reward functions
    grouped_rewards = rewards_tensor.sum(dim=1).view(num_prompts, num_generations)
    
    # Prompt-wise z-normalization (advantages)
    EPS = 1e-4
    grouped_advantages = (
        (grouped_rewards - grouped_rewards.mean(-1, keepdim=True)) / 
        (grouped_rewards.std(-1, keepdim=True) + EPS)
    )
    
    return grouped_advantages.flatten()


@dataclass
class CustomGRPOConfig:
    """Extended GRPO configuration with custom reward support."""
    
    # Reward configuration
    reward_config: RewardConfig = field(
        default_factory=RewardConfig,
        metadata={"help": "Configuration for reward functions"}
    )
    
    # Training parameters (keep existing verl parameters)
    total_epochs: int = field(
        default=15,
        metadata={"help": "Total number of training epochs"}
    )
    
    # Data parameters
    train_batch_size: int = field(
        default=128,
        metadata={"help": "Training batch size"}
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "Maximum prompt length"}
    )
    max_response_length: int = field(
        default=1024,
        metadata={"help": "Maximum response length"}
    )
    
    # Actor parameters
    actor_lr: float = field(
        default=5e-7,
        metadata={"help": "Actor learning rate"}
    )
    actor_ppo_mini_batch_size: int = field(
        default=64,
        metadata={"help": "PPO mini batch size"}
    )
    actor_ppo_micro_batch_size_per_gpu: int = field(
        default=4,
        metadata={"help": "PPO micro batch size per GPU"}
    )
    
    # Rollout parameters
    rollout_n: int = field(
        default=4,
        metadata={"help": "Number of rollout generations"}
    )
    rollout_tensor_model_parallel_size: int = field(
        default=4,
        metadata={"help": "Tensor model parallel size for rollout"}
    )
    rollout_gpu_memory_utilization: float = field(
        default=0.8,
        metadata={"help": "GPU memory utilization for rollout"}
    )
    
    # Algorithm parameters
    algorithm_adv_estimator: str = field(
        default="grpo",
        metadata={"help": "Advantage estimator algorithm"}
    )
    kl_coef: float = field(
        default=0.001,
        metadata={"help": "KL divergence coefficient"}
    )
    
    # Save and evaluation
    save_freq: int = field(
        default=10,
        metadata={"help": "Save frequency (in epochs)"}
    )
    test_freq: int = field(
        default=10,
        metadata={"help": "Test frequency (in epochs)"}
    )