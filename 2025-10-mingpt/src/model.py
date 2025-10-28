import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from yacs.config import CfgNode as CN


class GPT(nn.Module):

    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = "gpt-mini"
        C.vocab_size = 65
        C.block_size = 128
        C.d_model = 64
        C.num_layers = 8

        C.MLP = CN()
        C.MLP.hidden_size = 256

        C.Attention = CN()
        C.Attention.heads_size = 8

        return C.clone()

    def __init__(self, config):
        super().__init__()

        self.config = config
        vocab_size = self.config.vocab_size
        d_model = self.config.d_model
        block_size = self.config.block_size
        num_layers = self.config.num_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.embedding_position = nn.Embedding(block_size, d_model)
        torch.nn.init.normal_(self.embedding_position.weight, mean=0.0, std=0.02)
        self.register_buffer("position_ids", torch.arange(block_size))

        self.layers = nn.Sequential(*[GPT_Layer(config) for _ in range(num_layers)])

        self.softmax = nn.Softmax(dim=-1)
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x is a batch_size x seq_len tensor of token indices
        # y is a batch_size x seq_len tensor, the rows are one-off chars from x

        batch_size, seq_len = x.shape
        d_model = self.config.d_model

        # Ensure sequence length doesn't exceed block_size
        assert (
            seq_len <= self.config.block_size
        ), f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"

        token_embeddings = self.embedding(x) * (
            d_model**0.5
        )  # (batch_size, seq_len, d_model)
        pos_embeddings = self.embedding_position(
            self.position_ids[:seq_len]
        )  # (seq_len, d_model)
        embeddings = token_embeddings + pos_embeddings  # (batch_size, seq_len, d_model)

        hidden = self.layers(embeddings)

        logits = hidden @ self.embedding.weight.T

        loss = None
        if y is not None:
            loss = self.crossentropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss

    def configure_optimizers(self, config):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=0.0,
        )
        return optimizer

    def generate(self, x, n, temperature, do_sample, top_k):
        # x is a batch_size x seq_len tensor of token indices

        for _ in range(n):
            # Truncate context if it exceeds block_size (sliding window)
            x_cond = (
                x
                if x.size(1) <= self.config.block_size
                else x[:, -self.config.block_size :]
            )

            logits, _ = self.forward(x_cond)  # batch_size, seq_len, vocab_size
            logits = logits[:, -1]  # batch_size, vocab_size

            if top_k:
                v, _ = torch.topk(logits, k=top_k)  # batch_size, k
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = self.softmax(logits / temperature)

            if do_sample:
                token = torch.multinomial(probs, num_samples=1)  # batch_size, 1
            else:
                token = torch.argmax(probs, dim=-1, keepdim=True)  # batch_size, 1

            x = torch.cat([x, token], dim=1)

        return x


class GPT_Layer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        block_size = config.block_size
        d_model = config.d_model

        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.attention = GPT_Attention(config)
        self.mlp = GPT_MLP(config)

    def forward(self, x):
        # x is a batch_size x seq_len x d_model tensor

        block_attention = x + self.attention(self.ln_1(x))
        block_mlp = block_attention + self.mlp(self.ln_2(block_attention))

        return block_mlp


class GPT_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        block_size = config.block_size
        d_model = config.d_model
        heads_size = config.Attention.heads_size

        self.d_k = d_model // heads_size

        self.heads = nn.ModuleList(
            [GPT_Attention_Head(config) for _ in range(heads_size)]
        )

        self.W_o = nn.Parameter(torch.randn(heads_size * self.d_k, d_model) * 0.02)

    def forward(self, x):
        # x is a batch_size x seq_len x d_model tensor

        return torch.cat([head(x) for head in self.heads], dim=-1) @ self.W_o


class GPT_Attention_Head(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        block_size = config.block_size
        d_model = config.d_model
        heads_size = config.Attention.heads_size

        self.d_k = d_model // heads_size

        self.W_q = nn.Parameter(torch.randn(d_model, self.d_k) * 0.02)
        self.W_k = nn.Parameter(torch.randn(d_model, self.d_k) * 0.02)
        self.W_v = nn.Parameter(torch.randn(d_model, self.d_k) * 0.02)

        self.register_buffer(
            "mask",
            torch.triu(torch.full((block_size, block_size), float("-inf")), diagonal=1),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is a batch_size x seq_len x d_model tensor

        batch_size, seq_len, d_model = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        scores = Q @ K.transpose(-2, -1) / (self.d_k**0.5)
        # Use only the relevant portion of the mask for current sequence length
        A = self.softmax(
            scores + self.mask[:seq_len, :seq_len]
        )  # batch_size x seq_len x seq_len

        return A @ V  # batch_size x seq_len x d_k


class GPT_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        block_size = config.block_size
        d_model = config.d_model
        hidden_size = config.MLP.hidden_size

        self.fc1 = nn.Linear(d_model, hidden_size)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.fc1.bias)

        self.gelu = nn.GELU()

        self.fc2 = nn.Linear(hidden_size, d_model)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x is a batch_size x seq_len x d_model tensor

        return self.fc2(self.gelu(self.fc1(x)))
