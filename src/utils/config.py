"""
Configuration system for the SLM project.

Loads YAML configuration files, validates parameters per §51,
and provides nested attribute access.
"""

import copy
import math
import yaml
from pathlib import Path
from typing import Any, Optional


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """
    Hierarchical configuration object with attribute access.
    
    Loads from YAML files, supports nested access (config.model.hidden_size),
    defaults merging, and comprehensive validation.
    """
    
    def __init__(self, data: dict | None = None):
        self._data = data or {}
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)
        try:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def __repr__(self) -> str:
        return f"Config({self._data})"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default fallback."""
        try:
            value = self._data[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            return default
    
    def to_dict(self) -> dict:
        """Convert back to a plain dictionary (deep copy)."""
        return copy.deepcopy(self._data)
    
    def update(self, other: dict) -> None:
        """Deep-merge another dict into this config."""
        _deep_merge(self._data, other)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Config':
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config file must contain a YAML mapping, got {type(data)}")
        return cls(data)
    
    @classmethod
    def from_multiple(cls, *paths: str | Path) -> 'Config':
        """Load and merge multiple YAML files (later files override earlier)."""
        merged = {}
        for path in paths:
            cfg = cls.from_yaml(path)
            _deep_merge(merged, cfg._data)
        return cls(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base, modifying base in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_intermediate_size(hidden_size: int, intermediate_size: Any) -> int:
    """
    Resolve 'auto' intermediate_size for SwiGLU MLP.
    
    SwiGLU uses 3 projections (gate, up, down) instead of 2 (up, down) in standard MLP.
    To match the compute of a 4× GELU MLP, we use 8/3 × hidden_size ≈ 2.67×.
    Rounded up to the nearest multiple of 256 for memory alignment.
    """
    if intermediate_size == 'auto' or intermediate_size is None:
        raw = int(hidden_size * 8 / 3)
        return ((raw + 255) // 256) * 256
    return int(intermediate_size)


def validate_config(config: Config) -> None:
    """
    Validate the full configuration per §51.
    
    Fails loudly and specifically on any serious config error.
    """
    errors = []
    
    # --- Model validation ---
    if 'model' in config:
        m = config.model
        
        vocab_size = m.get('vocab_size', 0)
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            errors.append(f"model.vocab_size must be > 0, got {vocab_size}")
        
        hidden_size = m.get('hidden_size', 0)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            errors.append(f"model.hidden_size must be > 0, got {hidden_size}")
        
        num_query_heads = m.get('num_query_heads', 0)
        if not isinstance(num_query_heads, int) or num_query_heads <= 0:
            errors.append(f"model.num_query_heads must be > 0, got {num_query_heads}")
        
        num_kv_heads = m.get('num_kv_heads', 0)
        if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
            errors.append(f"model.num_kv_heads must be > 0, got {num_kv_heads}")
        
        if hidden_size > 0 and num_query_heads > 0:
            if hidden_size % num_query_heads != 0:
                errors.append(
                    f"model.hidden_size ({hidden_size}) must be divisible by "
                    f"model.num_query_heads ({num_query_heads})"
                )
        
        if num_query_heads > 0 and num_kv_heads > 0:
            if num_query_heads % num_kv_heads != 0:
                errors.append(
                    f"model.num_query_heads ({num_query_heads}) must be divisible by "
                    f"model.num_kv_heads ({num_kv_heads}) for GQA"
                )
        
        num_layers = m.get('num_layers', 0)
        if not isinstance(num_layers, int) or num_layers <= 0:
            errors.append(f"model.num_layers must be > 0, got {num_layers}")
        
        max_seq_len = m.get('max_seq_len', 0)
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            errors.append(f"model.max_seq_len must be > 0, got {max_seq_len}")
    
    # --- Training validation ---
    if 'training' in config:
        t = config.training
        
        micro_batch_size = t.get('micro_batch_size', 0)
        if not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
            errors.append(f"training.micro_batch_size must be > 0, got {micro_batch_size}")
        
        grad_accum = t.get('gradient_accumulation_steps', 0)
        if not isinstance(grad_accum, int) or grad_accum <= 0:
            errors.append(f"training.gradient_accumulation_steps must be > 0, got {grad_accum}")
        
        lr = t.get('learning_rate', 0)
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append(f"training.learning_rate must be > 0, got {lr}")
        
        warmup_ratio = t.get('warmup_ratio', 0)
        if not isinstance(warmup_ratio, (int, float)) or not (0 <= warmup_ratio <= 1):
            errors.append(f"training.warmup_ratio must be in [0, 1], got {warmup_ratio}")
        
        annealing_fraction = t.get('annealing_fraction', 0)
        if not isinstance(annealing_fraction, (int, float)) or not (0 <= annealing_fraction <= 1):
            errors.append(f"training.annealing_fraction must be in [0, 1], got {annealing_fraction}")
        
        schedule = t.get('schedule', 'wsd')
        if schedule not in ('wsd', 'cosine'):
            errors.append(f"training.schedule must be 'wsd' or 'cosine', got '{schedule}'")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(error_msg)


def validate_param_count(total_params: int) -> None:
    """Validate that parameter count is within the 65-85M target range."""
    if not (65_000_000 <= total_params <= 85_000_000):
        raise ConfigError(
            f"Parameter count {total_params:,} is outside the target range of 65-85M. "
            f"Adjust model configuration (layers, hidden_size, vocab_size) to hit the target."
        )


def get_effective_batch_size(config: Config) -> dict:
    """Compute and return effective batch size details."""
    t = config.training
    micro_batch = t.micro_batch_size
    grad_accum = t.gradient_accumulation_steps
    seq_len = t.get('max_seq_len', config.model.max_seq_len)
    
    effective_batch_tokens = micro_batch * grad_accum * seq_len
    return {
        'micro_batch_size': micro_batch,
        'gradient_accumulation_steps': grad_accum,
        'max_seq_len': seq_len,
        'effective_batch_tokens': effective_batch_tokens,
        'effective_batch_sequences': micro_batch * grad_accum,
    }
