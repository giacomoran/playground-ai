"""
Verify that model initialization produces expected loss values.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import torch

from model import GPT
from chatgpt import CharDataset, get_config
from utils import set_seed


def verify_init_loss():
    """Verify that the initial loss is close to -log(1/vocab_size)"""

    print("=" * 80)
    print("MODEL INITIALIZATION VERIFICATION")
    print("=" * 80)

    # Get config
    config = get_config()
    set_seed(config.system.seed)

    # Load data
    text = open("data/input.txt", "r").read()
    train_dataset = CharDataset(config.data, text)

    # Setup model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_block_size()

    print(f"Vocab size: {vocab_size}")
    print(f"Block size: {block_size}")
    print(f"Expected loss (uniform distribution): {math.log(vocab_size):.4f}")
    print()

    # Get a batch
    batch_size = 4
    indices = torch.randint(0, len(train_dataset), (batch_size,))
    x_batch = torch.stack([train_dataset[i][0] for i in indices])
    y_batch = torch.stack([train_dataset[i][1] for i in indices])

    print(f"Sample batch shape: x={x_batch.shape}, y={y_batch.shape}")
    print()

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, loss = model(x_batch, y_batch)
        print(f"Logits shape: {logits.shape}")
        print(f"Initial loss: {loss.item():.4f}")
        print()

        # Check if loss is reasonable
        expected_loss = -math.log(1 / vocab_size)
        diff = abs(loss.item() - expected_loss)

        print(f"Expected loss: {expected_loss}")
        print(f"Diff: {diff}")

    print("=" * 80)
    print()

    return model, train_dataset, config


if __name__ == "__main__":
    verify_init_loss()
