"""
Input-independent baseline experiments.

This module provides tools to train models with zero inputs to establish
a baseline that only learns the marginal token distribution.
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
    C.system.work_dir = "./out/input_independent"

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt-mini"

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = (
        5e-4  # the model we're using is so small that we can go a bit faster
    )
    C.trainer.max_iters = 1001

    return C.clone()


#:


class CharDataset(Dataset):
    """
    Emits batches of characters
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

        x = torch.zeros_like(x)  # zero all inputs

        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


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
        f"Training dataset: {len(train_dataset)} samples, block size: {train_dataset.get_block_size()}, vocab size: {train_dataset.get_vocab_size()}"
    )

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor(
                    [train_dataset.stoi[s] for s in context], dtype=torch.long
                )[None, ...].to(trainer.device)
                # Zero out inputs to match training (input-independent model)
                x = torch.zeros_like(x)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = "".join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback("on_batch_end", batch_end_callback)

    # run the optimization
    trainer.run()
