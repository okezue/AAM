from __future__ import annotations
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from aamemory.schema import SparseCode
def stablehashpair(i: int, j: int, seed: int) -> int:
    key = int(seed).to_bytes(8, byteorder="little", signed=False)
    digest = hashlib.blake2b(f"{i}:{j}".encode("ascii"), digest_size=8, key=key).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)
@dataclass(frozen=True)
class BoundAddress:
    code: SparseCode
    hidden_offset: int
    context_offset: int
    binding_offset: int
    neurogenesis_offset: int
    diagnostics: Mapping[str, int | float]
def addscaled(target: dict[int, float], index: int, value: float, scale: float) -> None:
    if value == 0.0 or scale == 0.0:
        return
    target[index] = target.get(index, 0.0) + float(value) * float(scale)
def composeboundaddress(
    hidden: SparseCode,
    context: SparseCode,
    *,
    bindingdimension: int = 8192,
    neurogenesis_reserved: int = 0,
    topk: int = 192,
    hidden_weight: float = 1.0,
    contextweight: float = 0.35,
    bindingweight: float = 0.45,
    seed: int = 17,
) -> BoundAddress:
    if bindingdimension < 0 or neurogenesis_reserved < 0:
        raise ValueError("binding and neurogenesis dimensions must be non-negative")
    hidden_offset = 0
    context_offset = hidden.dimension
    binding_offset = hidden.dimension + context.dimension
    neurogenesis_offset = binding_offset + bindingdimension
    total_dimension = neurogenesis_offset + neurogenesis_reserved
    values: dict[int, float] = {}
    for i, v in zip(hidden.indices, hidden.values, strict=True):
        addscaled(values, hidden_offset + i, v, hidden_weight)
    for i, v in zip(context.indices, context.values, strict=True):
        addscaled(values, context_offset + i, v, contextweight)
    if bindingdimension > 0 and hidden.indices and context.indices:
        hidden_pairs = list(zip(hidden.indices, hidden.values, strict=True))[: min(64, len(hidden.indices))]
        context_pairs = list(zip(context.indices, context.values, strict=True))[: min(32, len(context.indices))]
        for hi, hv in hidden_pairs:
            for ci, cv in context_pairs:
                bucket = stablehashpair(hi, ci, seed) % bindingdimension
                addscaled(values, binding_offset + bucket, math.copysign(abs(hv * cv), hv * cv), bindingweight)
    kept = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)[:topk]
    code = SparseCode.frommapping(total_dimension, dict(kept)).normalized()
    return BoundAddress(
        code=code,
        hidden_offset=hidden_offset,
        context_offset=context_offset,
        binding_offset=binding_offset,
        neurogenesis_offset=neurogenesis_offset,
        diagnostics={
            "hidden_nonzero": len(hidden.indices),
            "context_nonzero": len(context.indices),
            "bindingdimension": bindingdimension,
            "neurogenesis_reserved": neurogenesis_reserved,
            "address_nonzero": len(code.indices),
            "address_dimension": total_dimension,
        },
    )
