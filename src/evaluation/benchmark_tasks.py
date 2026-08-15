"""
Lightweight Evaluation Harness per §43.

Evaluates relative performance across checkpoints on standard tasks:
- Validation Perplexity
- LAMBADA completion perplexity / accuracy
- Synthetic / Sampled multiple-choice evaluation (HellaSwag / PIQA / ARC-Easy style)

NOTE per §43: At ~75M parameters, models will score near random chance (~25% on 4-way choice).
These tasks serve as relative signal between checkpoints, not absolute frontier claims.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List


class BenchmarkSuite:
    """Benchmark evaluation suite."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_lambada_sample(self, samples: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Evaluate log-likelihood / accuracy on completion samples (LAMBADA style).
        Each sample dict: {'context': str, 'target_word': str}
        """
        correct = 0
        total = 0
        total_loss = 0.0

        for sample in samples:
            context = sample["context"]
            target = sample["target_word"]
            full_text = context + " " + target

            ctx_ids = self.tokenizer.encode(context).ids
            full_ids = self.tokenizer.encode(full_text).ids

            if len(full_ids) <= len(ctx_ids):
                continue

            input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.device)
            target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=self.device)

            logits, _, _ = self.model(input_tensor)

            # Compute loss over target word tokens only
            target_start_pos = len(ctx_ids) - 1
            word_logits = logits[0, target_start_pos:]
            word_targets = target_tensor[0, target_start_pos:]

            loss = F.cross_entropy(word_logits, word_targets, reduction="mean")
            total_loss += loss.item()

            # Predict argmax
            preds = torch.argmax(word_logits, dim=-1)
            if torch.equal(preds, word_targets):
                correct += 1
            total += 1

        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        perplexity = math.exp(min(20.0, avg_loss))

        return {
            "lambada_accuracy": accuracy,
            "lambada_loss": avg_loss,
            "lambada_perplexity": perplexity,
        }

    @torch.no_grad()
    def evaluate_multiple_choice(self, questions: List[Dict[str, Any]], task_name: str = "arc_easy") -> Dict[str, float]:
        """
        Evaluate multiple-choice questions (HellaSwag / PIQA / ARC-Easy style).
        Each question dict: {'prompt': str, 'choices': [str, str, str, str], 'gold': int}
        """
        correct = 0
        total = 0

        for q in questions:
            prompt = q["prompt"]
            choices = q["choices"]
            gold = q["gold"]

            choice_losses = []

            for choice in choices:
                text = prompt + " " + choice
                prompt_ids = self.tokenizer.encode(prompt).ids
                full_ids = self.tokenizer.encode(text).ids

                if len(full_ids) <= len(prompt_ids):
                    choice_losses.append(float("inf"))
                    continue

                input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.device)
                target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=self.device)

                logits, _, _ = self.model(input_tensor)

                start_pos = len(prompt_ids) - 1
                choice_logits = logits[0, start_pos:]
                choice_targets = target_tensor[0, start_pos:]

                loss = F.cross_entropy(choice_logits, choice_targets, reduction="mean")
                choice_losses.append(loss.item())

            pred_choice = int(torch.argmin(torch.tensor(choice_losses)).item())
            if pred_choice == gold:
                correct += 1
            total += 1

        acc = correct / max(1, total)
        return {
            f"{task_name}_accuracy": acc,
            f"{task_name}_num_samples": total,
        }
