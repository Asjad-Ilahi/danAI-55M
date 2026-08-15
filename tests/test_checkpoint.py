"""
Unit tests for Checkpoint saving and loading.
"""

import tempfile
import unittest
from pathlib import Path
import torch

from src.utils.config import Config
from src.model.gpt import CausalLM
from src.model.ema import EMAModel
from src.training.optimizer import create_optimizer
from src.training.scheduler import WSDOrCosineScheduler
from src.training.checkpoint import CheckpointManager


class TestCheckpoint(unittest.TestCase):

    def test_checkpoint_save_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config.from_yaml("configs/debug.yaml")
            model = CausalLM(config)
            ema = EMAModel(model, decay=0.999)
            optim = create_optimizer(model)
            sched = WSDOrCosineScheduler(optim, max_steps=100)

            ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_dir)

            # Save checkpoint
            saved_path = ckpt_mgr.save_checkpoint(
                "test.pt",
                model=model,
                ema_model=ema,
                optimizer=optim,
                scheduler=sched,
                step=42,
                tokens_seen=10000,
                best_val_loss=1.234,
                config=config.to_dict(),
            )

            # Restore into new model
            new_model = CausalLM(config)
            new_ema = EMAModel(new_model, decay=0.999)
            new_optim = create_optimizer(new_model)
            new_sched = WSDOrCosineScheduler(new_optim, max_steps=100)

            restored_state = ckpt_mgr.load_checkpoint(
                saved_path,
                model=new_model,
                ema_model=new_ema,
                optimizer=new_optim,
                scheduler=new_sched,
            )

            self.assertEqual(restored_state["step"], 42)
            self.assertEqual(restored_state["tokens_seen"], 10000)
            self.assertAlmostEqual(restored_state["best_val_loss"], 1.234)

            # Verify model weights restored exactly
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
