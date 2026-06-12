"""Embedding model wrapper.

Loads jina-embeddings-v2-base-code with FP16 + Flash Attention 2.
Provides batch encoding: list of strings → (N, dim) numpy array.
"""

from typing import List

import numpy as np
from numpy.typing import NDArray
import torch
from tqdm import tqdm
from transformers import AutoModel

from utils.tokenizer import get_tokenizer


MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
BATCH_SIZE = 32
MAX_LENGTH = 512


class Embedder:
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Loading {MODEL_NAME}...")

        self.tokenizer = get_tokenizer()
        # flash_attention_2 is CUDA-only; fall back to eager on MPS/CPU
        attn = "flash_attention_2" if device == "cuda" else "eager"
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation=attn,
        )

        self.model = self.model.to(device)
        if device == "cuda":
            self.model = self.model.half()
            print("  Using FP16 + Flash Attention 2")
        elif device == "mps":
            # Apple GPUs deliver far higher FP16 than FP32 throughput
            # (M5: ~14 vs ~3.6 TFLOPS). Embeddings are direction-based and
            # L2-normalized, so FP16 is lossless here (cosine vs FP32 = 1.0000).
            self.model = self.model.half()
            print("  Using MPS (Apple Silicon) + FP16")

        self.model.eval()
        self.device = device
        self.dim: int = self.model.config.hidden_size

    def encode(self, texts: List[str]) -> NDArray[np.float32]:
        """Batch encode. Returns shape (len(texts), dim), float32.

        Length-bucketed batching: chunks are sorted by token length so each
        batch contains similar-length texts, then padded only to the batch's
        longest member (not a fixed 512). Short batches stay short, so the
        model wastes far less compute on padding. Results are scattered back
        to the original order. Measured ~+33% throughput on a real codebase
        (nginx) vs fixed max_length padding, with identical vectors.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Sort by token length; remember original positions to restore order.
        lengths = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])

        out = np.empty((len(texts), self.dim), dtype=np.float32)
        with torch.no_grad():
            for i in tqdm(range(0, len(order), BATCH_SIZE), desc="encoding", leave=False):
                idx = order[i : i + BATCH_SIZE]
                batch = [texts[j] for j in idx]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="longest",  # pad to this batch's max, not a fixed 512
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # Masked mean pooling: average only over real tokens, not the
                # padding added to equalize batch length. A plain .mean(dim=1)
                # would average padding positions in too, so the same text
                # yields different vectors depending on what it was batched
                # with — breaking the index/query symmetry (queries are
                # encoded one-at-a-time with no padding; index chunks are
                # batched with padding).
                hidden = outputs.last_hidden_state  # (B, T, dim)
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
                summed = (hidden * mask).sum(dim=1)  # (B, dim) — padding zeroed out
                counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1) — real token count
                embs = (summed / counts).float().cpu().numpy()
                # L2-normalize so sqlite-vec's L2 distance == cosine distance,
                # giving stable [0, 2] scores comparable across queries.
                embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
                # Scatter back to original positions.
                for k, j in enumerate(idx):
                    out[j] = embs[k]
        return out

