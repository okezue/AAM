from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.encoding.projection import GaussianRandomProjector
from aamemory.schema import SparseCode
class SentenceTransformerFeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        *,
        modelname: str = "sentence-transformers/all-MiniLM-L6-v2",
        outputdimension: int = 32768,
        topk: int = 128,
        seed: int = 0,
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerFeatureEncoder requires `pip install -e .[embeddings]`"
            ) from exc
        self.model = SentenceTransformer(
            modelname, device=device, trust_remote_code=trust_remote_code
        )
        dense_dimension = int(self.model.get_sentence_embedding_dimension())
        self.projector = GaussianRandomProjector(dense_dimension, outputdimension, seed=seed)
        self.topk = int(topk)
        self.modelname = modelname
    @property
    def dimension(self) -> int:
        return self.projector.outputdimension
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        dense = np.asarray(
            self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0],
            dtype=np.float32,
        )
        projected = self.projector(dense)
        code = SparseCode.fromdense(projected, topk=self.topk, normalize=True)
        return EncodingResult(
            code=code,
            payload={"dense_embedding": dense.tolist()},
            diagnostics={"encoder": "sentence_transformer", "model": self.modelname},
        )
