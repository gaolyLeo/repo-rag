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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_NAME}...")

        self.tokenizer = get_tokenizer()
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        self.model = self.model.to(device)
        if device == "cuda":
            self.model = self.model.half()
            print("  Using FP16 + Flash Attention 2")

        self.model.eval()
        self.device = device
        self.dim: int = self.model.config.hidden_size

    def encode(self, texts: List[str]) -> NDArray[np.float32]:
        """Batch encode. Returns shape (len(texts), dim), float32."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        out = np.empty((len(texts), self.dim), dtype=np.float32)
        pos = 0
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encoding", leave=False):
                batch = texts[i : i + BATCH_SIZE]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # Mean pooling over sequence dim, then fp16 → fp32 for index
                embs = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
                # L2-normalize so sqlite-vec's L2 distance == cosine distance,
                # giving stable [0, 2] scores comparable across queries.
                embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
                out[pos : pos + len(batch)] = embs
                pos += len(batch)
        return out

