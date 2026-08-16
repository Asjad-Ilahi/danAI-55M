"""
Main training entrypoint script per §44 & §15.

Handles experiment directory setup (experiments/exp_NNN/), configuration loading,
dataset initialization, and starting the Trainer loop.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import Config
from src.data.shard_dataset import ShardDataset
from src.training.trainer import Trainer


def get_next_experiment_dir(base_dir: Path = Path("experiments")) -> Path:
    """Find next non-overwriting experiment directory (experiments/exp_001, exp_002, ...)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = list(base_dir.glob("exp_*"))
    indices = []
    for p in existing:
        try:
            idx = int(p.name.split("_")[1])
            indices.append(idx)
        except (IndexError, ValueError):
            continue
    next_idx = max(indices, default=0) + 1
    exp_dir = base_dir / f"exp_{next_idx:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train 75M SLM model from scratch")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--train-config", type=str, default="configs/train.yaml", help="Path to train config")
    parser.add_argument("--data-dir", type=str, default="data/shards/train", help="Path to training shards")
    parser.add_argument("--val-data-dir", type=str, default="data/shards/val", help="Path to validation shards")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--debug", action="store_true", help="Use debug config (tiny model for fast testing)")
    args = parser.parse_args()

    # Load and merge configurations
    if args.debug:
        print("\n*** DEBUG MODE ACTIVE — Using configs/debug.yaml ***\n")
        config = Config.from_yaml("configs/debug.yaml")
    else:
        model_cfg = Config.from_yaml(args.config)
        train_cfg = Config.from_yaml(args.train_config)
        config = Config.from_multiple(args.config, args.train_config)

    # Set up non-overwriting experiment directory (§44)
    exp_dir = get_next_experiment_dir()
    print(f"Experiment directory created: {exp_dir}")

    # Load datasets
    train_data_path = Path(args.data_dir)
    val_data_path = Path(args.val_data_dir)

    pack_mask = config.training.get("pack_with_document_mask", True)
    seq_len = config.model.max_seq_len

    print(f"Loading training dataset from {train_data_path}...")
    train_dataset = ShardDataset(train_data_path, seq_len=seq_len, pack_with_document_mask=pack_mask)

    val_dataset = None
    if val_data_path.exists() and list(val_data_path.glob("shard_*.bin")):
        print(f"Loading validation dataset from {val_data_path}...")
        val_dataset = ShardDataset(val_data_path, seq_len=seq_len, pack_with_document_mask=pack_mask)

    # Initialize trainer and start training
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        experiment_dir=exp_dir,
        resume_checkpoint=args.resume,
    )

    trainer.train()


if __name__ == "__main__":
    main()
