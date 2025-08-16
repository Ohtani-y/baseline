"""
GRPO Reward Functions for verl
Ported from Kaggle Fast-Math-R1 implementation
"""

import re
import math
from typing import List, Union, Dict, Any
from Levenshtein import ratio as levenshtein_ratio
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def extract_contents(completions: Union[List[Dict[str, str]], List[str]]) -> List[str]:
    """Extract content from completions."""
    if not completions:
        return []
    
    if isinstance(completions[0], dict):
        contents = [completion['content'] if 'content' in completion else str(completion) 
                   for completion in completions]
    else:
        contents = completions
    return contents


def extract_answer_after_think(text: str) -> str:
    """Extract answer text that appears after the last </think> tag."""
    # Find the last </think> tag and extract everything after it
    last_think_pos = text.rfind('</think>')
    if last_think_pos == -1:
        return ""
    
    # Extract text after the last </think> tag
    after_think = text[last_think_pos + len('</think>'):]
    
    # Look for newline followed by content
    pattern = r'\s*\n\s*(.*?)(?:\s*$)'
    match = re.search(pattern, after_think, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer
    return ""


def check_answer(completion: str, answer: str) -> float:
    """Check if the completion matches the ground truth answer using math-verify."""
    # Extract the answer from completion (after </think> tag)
    extracted_answer = extract_answer_after_think(completion)
    if not extracted_answer:
        return 0.0
    
    # Parse ground truth answer
    gold_parsed = parse('\\boxed{' + answer + '}')
    if len(gold_parsed) != 0:
        # Wrap extracted answer in \boxed{} for verification
        answer_to_verify = '\\boxed{' + extracted_answer + '}'
        answer_parsed = parse(
            answer_to_verify,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
            reward = 0.0
    else:
        reward = 1.0
    return reward


def format_reward2(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that checks if an answer appears after </think> tag.
    This encourages the model to provide the final answer after reasoning.
    
    Returns:
        List of rewards (1.0 if format is correct, 0.0 otherwise)
    """
    rewards = []
    for content in extract_contents(completions):
        # Check if there's a </think> tag followed by an answer
        pattern = r'</think>\s*\n\s*(.+)'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        if match and match.group(1).strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def cosine_scaled_reward(
    completions: List[str],
    answer: List[str],
    max_value_correct: float = 1.0,
    min_value_correct: float = 0.1,
    max_value_wrong: float = -0.1,
    min_value_wrong: float = -1.0,
    max_len: int = 30000,
    clip_len: bool = True,
    **kwargs
) -> List[float]:
    """
    Reward function that scales based on completion length using a cosine schedule.
    
    - Shorter correct solutions are rewarded more than longer ones
    - Longer incorrect solutions are penalized less than shorter ones
    
    Args:
        completions: List of model completions
        answer: List of ground truth answers
        max_value_correct: Maximum reward for correct answers (short)
        min_value_correct: Minimum reward for correct answers (long)
        max_value_wrong: Maximum reward for wrong answers (long)
        min_value_wrong: Minimum reward for wrong answers (short)
        max_len: Maximum length for scaling
        clip_len: Whether to clip progress to [0, 1]
    
    Returns:
        List of scaled rewards
    """
    rewards = []
    
    for content, gt in zip(extract_contents(completions), answer):
        is_correct = check_answer(content, str(gt))
        gen_len = len(content)
        
        # Apply cosine scaling based on length
        progress = gen_len / max_len
        if clip_len:
            progress = min(1.0, progress)
        cosine = math.cos(progress * math.pi)
        
        if is_correct > 0:  # Since check_answer returns float (0.0 or 1.0)
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong
        
        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        rewards.append(float(reward))
    
    return rewards


def length_reward(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """
    Compute length-based rewards to discourage overthinking and promote token efficiency.
    
    Args:
        completions: List of model completions
        answer: List of ground truth answers
    
    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = extract_contents(completions)
    correctness = [check_answer(content, str(gt)) for content, gt in zip(contents, answer)]
    
    # Calculate lengths
    lengths = [len(content) for content in contents]
    if not lengths:
        return []
        
    min_len = min(lengths)
    max_len = max(lengths)
    
    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)
    
    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
        
        if is_correct > 0:  # Since check_answer returns float (0.0 or 1.0)
            reward = lambda_val
        else:
            reward = min(0, lambda_val)
        
        rewards.append(float(reward))
    
    return rewards


def accuracy_reward(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """
    Simple accuracy reward function.
    
    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    return [check_answer(content, str(gt)) for content, gt in zip(extract_contents(completions), answer)]


# Registry for easy access to reward functions
REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format2": format_reward2,
    "cosine": cosine_scaled_reward,
    "length": length_reward,
}


def get_cosine_scaled_reward(
    max_value_correct: float = 1.0,
    min_value_correct: float = 0.1,
    max_value_wrong: float = -0.1,
    min_value_wrong: float = -1.0,
    max_len: int = 30000,
    clip_len: bool = True,
):
    """
    Factory function to create a cosine scaled reward function with specific parameters.
    """
    def reward_func(completions: List[str], answer: List[str], **kwargs) -> List[float]:
        return cosine_scaled_reward(
            completions, answer,
            max_value_correct=max_value_correct,
            min_value_correct=min_value_correct,
            max_value_wrong=max_value_wrong,
            min_value_wrong=min_value_wrong,
            max_len=max_len,
            clip_len=clip_len,
            **kwargs
        )
    return reward_func