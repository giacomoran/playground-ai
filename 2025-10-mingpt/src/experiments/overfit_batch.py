"""
Overfit a single batch experiment.

This module trains a model on a tiny dataset (just a few examples) with
increased capacity to verify the model can achieve zero loss. This is a
critical sanity check - if the model can't overfit a tiny batch, there's
likely a bug in the implementation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode as CN

from model import GPT
from trainer import Trainer
from utils import set_seed, setup_logging, merge_from_args


#:


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/overfit_batch"

    # data
    C.data = CharDataset.get_default_config()

    # model - use larger capacity to ensure we can overfit
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt-nano"  # We'll use more layers than mini
    C.model.n_layer = 6  # Increase layers for more capacity
    C.model.n_head = 6  # Increase attention heads
    C.model.n_embd = 192  # Increase embedding dimension

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    C.trainer.max_iters = 2001  # More iterations to ensure we reach zero loss
    C.trainer.batch_size = 4  # Small batch size to match our tiny dataset

    return C.clone()


#:


class CharDataset(Dataset):
    """
    Emits only a tiny subset of examples for overfitting experiments.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        C.num_examples = 4  # Only use 4 examples total
        return C.clone()

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("Full data has %d characters, %d unique." % (data_size, vocab_size))
        print("Unique characters:", "".join(chars))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

        # Pre-compute the tiny dataset
        self._precompute_examples()

    def _precompute_examples(self):
        """Precompute a fixed tiny set of examples."""
        num_examples = self.config.num_examples
        self.examples = []

        # Take examples from different parts of the dataset
        full_length = len(self.data) - self.config.block_size
        step = full_length // num_examples

        for i in range(num_examples):
            idx = i * step
            chunk = self.data[idx : idx + self.config.block_size + 1]
            dix = [self.stoi[s] for s in chunk]
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
            self.examples.append((x, y))

        print(f"\nOverfit dataset: Using only {len(self.examples)} examples")
        print("Example sequences:")
        for i, (x, y) in enumerate(self.examples):
            text = "".join(
                [self.itos[int(idx)] for idx in x[:50]]
            )  # Show first 50 chars
            print(f"  Example {i}: {repr(text)}...")

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Always return from our tiny fixed set
        return self.examples[idx % len(self.examples)]


#:

if __name__ == "__main__":

    # get default config and overrides from the command line, if any
    config = get_config()
    merge_from_args(config, sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open(
        "data/input.txt", "r"
    ).read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # Print summary of the dataset
    print(
        f"\nTraining dataset: {len(train_dataset)} samples, block size: {train_dataset.get_block_size()}, vocab size: {train_dataset.get_vocab_size()}"
    )

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # Print model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

        if (
            trainer.iter_num % 200 == 0
            or trainer.iter_num == trainer.config.max_iters - 1
        ):
            print("\n" + "=" * 80)
            print(f"EVALUATION AT ITERATION {trainer.iter_num}")
            print("=" * 80)

            model.eval()
            with torch.no_grad():
                # Evaluate on all our tiny training examples
                total_loss = 0
                for i, (x, y) in enumerate(train_dataset.examples):
                    x_batch = x.unsqueeze(0).to(trainer.device)
                    y_batch = y.unsqueeze(0).to(trainer.device)

                    # Get model predictions
                    logits, loss = model(x_batch, y_batch)
                    total_loss += loss.item()

                    # Get predicted tokens
                    pred_tokens = logits.argmax(dim=-1)[0]

                    # Convert to text
                    input_text = "".join([train_dataset.itos[int(idx)] for idx in x])
                    target_text = "".join([train_dataset.itos[int(idx)] for idx in y])
                    pred_text = "".join(
                        [train_dataset.itos[int(idx)] for idx in pred_tokens]
                    )

                    # Calculate accuracy
                    correct = (pred_tokens == y_batch[0]).sum().item()
                    total = len(y_batch[0])
                    accuracy = 100 * correct / total

                    print(
                        f"\nExample {i} (loss: {loss.item():.5f}, accuracy: {accuracy:.1f}%):"
                    )
                    print(f"  Input:  {repr(input_text[:80])}...")
                    print(f"  Target: {repr(target_text[:80])}...")
                    print(f"  Pred:   {repr(pred_text[:80])}...")

                    # Show mismatches
                    if accuracy < 100:
                        mismatches = []
                        for j, (t, p) in enumerate(
                            zip(target_text[:80], pred_text[:80])
                        ):
                            if t != p:
                                mismatches.append(f"pos {j}: '{t}' -> '{p}'")
                        if mismatches:
                            print(
                                f"  Mismatches (first 5): {', '.join(mismatches[:5])}"
                            )

                avg_loss = total_loss / len(train_dataset.examples)
                print(
                    f"\nAverage loss across {len(train_dataset.examples)} examples: {avg_loss:.5f}"
                )

                # Also generate from scratch to see what the model has learned
                print("\n" + "-" * 80)
                print("Generating sample text:")
                # Start with zeros or the first token
                context = torch.zeros((1, 1), dtype=torch.long).to(trainer.device)
                y = model.generate(
                    context, 200, temperature=1.0, do_sample=True, top_k=10
                )[0]
                completion = "".join([train_dataset.itos[int(i)] for i in y])
                print(repr(completion))
                print("-" * 80 + "\n")

            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback("on_batch_end", batch_end_callback)

    print("\n" + "=" * 80)
    print("STARTING OVERFITTING EXPERIMENT")
    print("=" * 80)
    print(f"Goal: Achieve near-zero loss on {len(train_dataset)} examples")
    print(f"Model capacity: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("=" * 80 + "\n")

    # run the optimization
    trainer.run()

    print("\n" + "=" * 80)
    print("OVERFITTING EXPERIMENT COMPLETE")
    print("=" * 80)
