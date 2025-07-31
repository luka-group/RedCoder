from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class AttackConfig:
    """Configuration for the jailbreak attack."""
    max_turns: int = 5
    max_attempts: int = 10
    max_new_tokens: int = 256
    temperature: float = 0.7
    gpt_temperature: float = 1.2
    library_name: str = "prototype_conversation"


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    user_message: str
    assistant_response: str


# Additional hyperparameters that might be useful
@dataclass
class ModelConfig:
    """Configuration for model settings."""
    victim_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    guardrail_model_path: str = "Rakancorle1/amz_safe"
    dtype: str = "float16"  # or "float32", "bfloat16"


@dataclass
class PathConfig:
    """Configuration for file paths."""
    tasks_file: str = "../data/gaming_cwe.txt"
    base_output_path: str = "results"


@dataclass
class SafetyConfig:
    """Configuration for safety and guardrail settings."""
    max_safety_tokens: int = 100
    safety_pad_token_id: int = 0 