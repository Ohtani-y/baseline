"""
Custom verl trainer with Kaggle reward functions integration
"""

import os
import sys
import json
import torch
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import hydra

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl_grpo_rewards import REWARD_FUNCS_REGISTRY, get_cosine_scaled_reward
from verl_grpo_config import RewardConfig, compute_combined_rewards


def create_reward_function(config: DictConfig):
    """
    Create a composite reward function based on configuration.
    
    Args:
        config: Hydra configuration containing reward settings
    
    Returns:
        A reward function that computes combined rewards
    """
    # Extract reward configuration
    reward_funcs = config.get('reward', {}).get('funcs', ['accuracy'])
    reward_weights = config.get('reward', {}).get('weights', None)
    
    # Set default weights if not provided
    if reward_weights is None:
        reward_weights = [1.0] * len(reward_funcs)
    
    # Validate weights match number of functions
    if len(reward_weights) != len(reward_funcs):
        raise ValueError(
            f"Number of reward weights ({len(reward_weights)}) must match "
            f"number of reward functions ({len(reward_funcs)})"
        )
    
    # Create reward function instances
    reward_instances = []
    for func_name in reward_funcs:
        if func_name == 'cosine':
            # Create cosine reward with specific parameters
            cosine_params = {
                'max_value_correct': config.get('reward', {}).get('cosine_max_value_correct', 1.0),
                'min_value_correct': config.get('reward', {}).get('cosine_min_value_correct', 0.1),
                'max_value_wrong': config.get('reward', {}).get('cosine_max_value_wrong', -0.1),
                'min_value_wrong': config.get('reward', {}).get('cosine_min_value_wrong', -1.0),
                'max_len': config.get('reward', {}).get('cosine_max_len', 30000),
                'clip_len': config.get('reward', {}).get('cosine_clip_len', True),
            }
            reward_instances.append(get_cosine_scaled_reward(**cosine_params))
        elif func_name in REWARD_FUNCS_REGISTRY:
            reward_instances.append(REWARD_FUNCS_REGISTRY[func_name])
        else:
            raise ValueError(f"Unknown reward function: {func_name}")
    
    def combined_reward_function(
        prompts: List[str],
        completions: List[str],
        answers: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute combined rewards using multiple reward functions.
        
        This implements the Kaggle approach:
        1. Compute rewards from each function
        2. Apply weighted sum
        3. Normalize per prompt (z-normalization)
        
        Args:
            prompts: List of prompts (repeated for multiple generations)
            completions: List of model completions
            answers: List of ground truth answers (repeated for multiple generations)
        
        Returns:
            List of combined and normalized rewards (advantages)
        """
        # Compute rewards from each function
        rewards_list = []
        for reward_func in reward_instances:
            rewards = reward_func(completions, answers, **kwargs)
            rewards_list.append(rewards)
        
        # Determine number of unique prompts and generations
        unique_prompts = []
        seen = set()
        for p in prompts:
            if p not in seen:
                unique_prompts.append(p)
                seen.add(p)
        
        num_prompts = len(unique_prompts)
        num_generations = len(prompts) // num_prompts
        
        # Compute combined rewards with normalization
        advantages = compute_combined_rewards(
            rewards_list, 
            reward_weights,
            num_prompts,
            num_generations
        )
        
        return advantages.tolist()
    
    # Store metadata for logging
    combined_reward_function.reward_funcs = reward_funcs
    combined_reward_function.reward_weights = reward_weights
    
    return combined_reward_function


def patch_verl_trainer_for_custom_rewards(config: DictConfig):
    """
    Patch verl trainer to use custom reward functions.
    
    This function modifies the verl PPO trainer to use our custom reward computation
    while maintaining all other aspects of the training loop.
    
    Args:
        config: Hydra configuration
    """
    # Import verl modules
    try:
        from verl.trainer.ppo import PPOTrainer
        from verl.trainer.ppo.reward import RewardFunction
    except ImportError:
        print("Warning: verl modules not found. Please ensure verl is installed.")
        return None
    
    # Create custom reward function
    custom_reward_fn = create_reward_function(config)
    
    # Create a wrapper class that verl can use
    class CustomRewardFunction(RewardFunction):
        """Wrapper to integrate custom rewards into verl."""
        
        def __init__(self):
            super().__init__()
            self.reward_fn = custom_reward_fn
            self.reward_funcs = custom_reward_fn.reward_funcs
            self.reward_weights = custom_reward_fn.reward_weights
        
        def compute_rewards(
            self,
            prompts: List[str],
            completions: List[str],
            answers: Optional[List[str]] = None,
            **kwargs
        ) -> torch.Tensor:
            """
            Compute rewards using our custom function.
            
            Returns:
                Tensor of rewards/advantages
            """
            if answers is None:
                # Try to extract answers from kwargs or data
                answers = kwargs.get('ground_truth', kwargs.get('answer', None))
                if answers is None:
                    raise ValueError("No ground truth answers provided for reward computation")
            
            # Compute rewards using our custom function
            rewards = self.reward_fn(prompts, completions, answers, **kwargs)
            
            # Convert to tensor
            return torch.tensor(rewards, dtype=torch.float32)
        
        def get_reward_info(self) -> Dict[str, Any]:
            """Get information about the reward configuration."""
            return {
                'reward_functions': self.reward_funcs,
                'reward_weights': self.reward_weights,
                'type': 'custom_combined_rewards'
            }
    
    return CustomRewardFunction()


@hydra.main(version_base=None, config_path=".", config_name="grpo_config")
def main(cfg: DictConfig):
    """
    Main entry point for custom verl training with Kaggle rewards.
    
    This function:
    1. Sets up the custom reward function
    2. Initializes verl trainer with custom rewards
    3. Runs training with existing epoch and rollout settings
    """
    print("=" * 80)
    print("Starting verl training with custom Kaggle reward functions")
    print("=" * 80)
    
    # Print reward configuration
    print("\nReward Configuration:")
    print(f"  Functions: {cfg.get('reward', {}).get('funcs', ['accuracy'])}")
    print(f"  Weights: {cfg.get('reward', {}).get('weights', 'default (1.0 for all)')}")
    
    if 'cosine' in cfg.get('reward', {}).get('funcs', []):
        print("\nCosine Reward Parameters:")
        print(f"  max_value_correct: {cfg.get('reward', {}).get('cosine_max_value_correct', 1.0)}")
        print(f"  min_value_correct: {cfg.get('reward', {}).get('cosine_min_value_correct', 0.1)}")
        print(f"  max_value_wrong: {cfg.get('reward', {}).get('cosine_max_value_wrong', -0.1)}")
        print(f"  min_value_wrong: {cfg.get('reward', {}).get('cosine_min_value_wrong', -1.0)}")
        print(f"  max_len: {cfg.get('reward', {}).get('cosine_max_len', 30000)}")
        print(f"  clip_len: {cfg.get('reward', {}).get('cosine_clip_len', True)}")
    
    print("\nTraining Parameters (unchanged):")
    print(f"  Total epochs: {cfg.get('trainer', {}).get('total_epochs', 15)}")
    print(f"  Rollout n: {cfg.get('actor_rollout_ref', {}).get('rollout', {}).get('n', 4)}")
    print("=" * 80)
    
    # Get custom reward function
    custom_reward = patch_verl_trainer_for_custom_rewards(cfg)
    
    if custom_reward is None:
        print("Warning: Could not create custom reward function. Using default verl trainer.")
        # Fall back to standard verl training
        from verl.trainer.main_ppo import main as verl_main
        return verl_main(cfg)
    
    # Import and modify verl trainer
    try:
        from verl.trainer.main_ppo import main as verl_main
        from verl.trainer.main_ppo import make_ppo_trainer
        
        # Monkey-patch the trainer creation to use our custom reward
        original_make_trainer = make_ppo_trainer
        
        def custom_make_trainer(*args, **kwargs):
            """Create trainer with custom reward function."""
            trainer = original_make_trainer(*args, **kwargs)
            # Replace the reward function
            if hasattr(trainer, 'reward_function'):
                trainer.reward_function = custom_reward
            return trainer
        
        # Replace the function
        import verl.trainer.main_ppo
        verl.trainer.main_ppo.make_ppo_trainer = custom_make_trainer
        
        # Run training with modified configuration
        return verl_main(cfg)
        
    except ImportError as e:
        print(f"Error importing verl modules: {e}")
        print("Please ensure verl is properly installed.")
        return None


if __name__ == "__main__":
    main()