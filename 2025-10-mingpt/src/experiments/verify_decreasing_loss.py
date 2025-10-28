"""
Verify decreasing training loss experiment.

This experiment verifies that:
1. A toy model with limited capacity shows underfitting (loss plateaus at a high value)
2. Increasing model capacity leads to lower training loss
3. The training dynamics are stable and loss decreases properly

This is a critical sanity check to ensure the training loop and model are working correctly.
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


def get_config(model_size="tiny"):
    """
    Get configuration for different model sizes.

    Args:
        model_size: "tiny" (underfitting), "small" (better capacity), or "medium" (even better)
    """
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = f"./out/verify_loss_{model_size}"

    # data
    C.data = CharDataset.get_default_config()

    # model - configure based on size
    C.model = GPT.get_default_config()

    if model_size == "tiny":
        # Very small model - should show underfitting
        C.model.model_type = "gpt-tiny"
        C.model.d_model = 32
        C.model.num_layers = 2
        C.model.MLP.hidden_size = 64
        C.model.Attention.heads_size = 4
    elif model_size == "small":
        # Slightly larger - should perform better
        C.model.model_type = "gpt-small"
        C.model.d_model = 64
        C.model.num_layers = 4
        C.model.MLP.hidden_size = 128
        C.model.Attention.heads_size = 4
    elif model_size == "medium":
        # Even larger - should perform even better
        C.model.model_type = "gpt-medium"
        C.model.d_model = 128
        C.model.num_layers = 6
        C.model.MLP.hidden_size = 256
        C.model.Attention.heads_size = 8
    else:
        raise ValueError(f"Unknown model_size: {model_size}")

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    C.trainer.max_iters = 1001  # Train for 1000 iterations
    C.trainer.batch_size = 32  # Use reasonable batch size

    return C.clone()


#:


class CharDataset(Dataset):
    """
    Character-level dataset from chatgpt.py
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C.clone()

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))
        print("Unique characters:", "".join(chars))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


#:


def run_experiment(model_size="tiny"):
    """Run training for a specific model size."""

    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT WITH MODEL SIZE: {model_size.upper()}")
    print("=" * 80)

    # get default config
    config = get_config(model_size)
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open("data/input.txt", "r").read()
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

    # Track loss history
    loss_history = []

    # iteration callback
    def batch_end_callback(trainer):
        # Record loss at every iteration
        loss_history.append(trainer.loss.item())

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

        if trainer.iter_num % 200 == 0:
            # Show some generation samples
            model.eval()
            with torch.no_grad():
                # sample from the model
                context = "O God, O God!"
                x = torch.tensor(
                    [train_dataset.stoi[s] for s in context], dtype=torch.long
                )[None, ...].to(trainer.device)
                y = model.generate(x, 200, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = "".join([train_dataset.itos[int(i)] for i in y])
                print("\n" + "-" * 80)
                print("Sample generation:")
                print(completion)
                print("-" * 80 + "\n")
            model.train()

        # Save at the end
        if trainer.iter_num == trainer.config.max_iters - 1:
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)

    trainer.set_callback("on_batch_end", batch_end_callback)

    print(f"\nStarting training with {model_size} model...")
    print(f"Model capacity: {n_params:,} parameters")
    print("=" * 80 + "\n")

    # run the optimization
    trainer.run()

    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE FOR {model_size.upper()} MODEL")
    print("=" * 80)
    print(f"Initial loss: {loss_history[0]:.5f}")
    print(f"Final loss: {loss_history[-1]:.5f}")
    print(f"Loss reduction: {loss_history[0] - loss_history[-1]:.5f}")

    # Calculate average loss over last 100 iterations
    avg_final_loss = sum(loss_history[-100:]) / len(loss_history[-100:])
    print(f"Average final loss (last 100 iters): {avg_final_loss:.5f}")

    # Check if loss is still decreasing (compare last 100 vs previous 100)
    avg_mid_loss = sum(loss_history[-200:-100]) / 100
    if avg_mid_loss > avg_final_loss:
        print(f"✓ Loss still decreasing (prev 100 avg: {avg_mid_loss:.5f})")
    else:
        print(f"⚠ Loss plateaued (prev 100 avg: {avg_mid_loss:.5f})")

    print("=" * 80 + "\n")

    return {
        "model_size": model_size,
        "n_params": n_params,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "avg_final_loss": avg_final_loss,
        "loss_history": loss_history,
    }


#:


if __name__ == "__main__":

    # Parse command line arguments
    if len(sys.argv) > 1:
        model_size = sys.argv[1]
        result = run_experiment(model_size)
    else:
        # Run all three experiments in sequence
        print("\n" + "=" * 80)
        print("RUNNING FULL EXPERIMENT: VERIFY DECREASING TRAINING LOSS")
        print("=" * 80)
        print("This will train models of increasing capacity and verify that")
        print("training loss decreases as model capacity increases.")
        print("=" * 80 + "\n")

        results = []
        for model_size in ["tiny", "small", "medium"]:
            result = run_experiment(model_size)
            results.append(result)

        # Print final comparison
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE - LOSS COMPARISON")
        print("=" * 80)
        print(
            f"{'Model Size':<12} {'Parameters':<15} {'Initial Loss':<15} {'Final Loss':<15} {'Reduction':<15}"
        )
        print("-" * 80)
        for r in results:
            reduction = r["initial_loss"] - r["final_loss"]
            print(
                f"{r['model_size']:<12} {r['n_params']:<15,} {r['initial_loss']:<15.5f} {r['final_loss']:<15.5f} {reduction:<15.5f}"
            )

        print("-" * 80)
        print("\nVERIFICATION RESULTS:")

        # Check that final loss decreases with capacity
        print("\n1. Does final loss decrease with increased capacity?")
        for i in range(len(results) - 1):
            curr = results[i]
            next_model = results[i + 1]
            improvement = curr["avg_final_loss"] - next_model["avg_final_loss"]
            if improvement > 0:
                print(
                    f"   ✓ {curr['model_size']} → {next_model['model_size']}: loss improved by {improvement:.5f}"
                )
            else:
                print(
                    f"   ✗ {curr['model_size']} → {next_model['model_size']}: loss got worse by {-improvement:.5f}"
                )

        print("\n2. Model capacity vs final performance:")
        tiny_loss = results[0]["avg_final_loss"]
        small_loss = results[1]["avg_final_loss"]
        medium_loss = results[2]["avg_final_loss"]

        if tiny_loss > small_loss > medium_loss:
            print("   ✓ Clear capacity-performance relationship observed!")
            print("   ✓ Tiny model shows underfitting (highest loss)")
            print("   ✓ Increasing capacity consistently reduces training loss")
        else:
            print("   ⚠ Unexpected capacity-performance relationship")

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
