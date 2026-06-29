from __future__ import annotations
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from aamemory.schema import SourceRef, SparseCode, utcnowiso
EngramStatus = Literal["observed", "corrected", "superseded", "hypothetical", "rejected"]
@dataclass
class EngramVersion:
    memory_id: str
    version_id: str
    status: EngramStatus = "observed"
    parent_version_id: str | None = None
    supersedes: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    confidence: float = 1.0
    authority: float = 1.0
    source_checksum: str | None = None
    created_at: str = field(default_factory=utcnowiso)
    last_verified_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def todict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = dict(self.metadata)
        return data
    @classmethod
    def fromdict(cls, value: Mapping[str, Any]) -> EngramVersion:
        return cls(
            memory_id=str(value["memory_id"]),
            version_id=str(value["version_id"]),
            status=str(value.get("status", "observed")),
            parent_version_id=value.get("parent_version_id"),
            supersedes=tuple(str(x) for x in value.get("supersedes", ())),
            contradicts=tuple(str(x) for x in value.get("contradicts", ())),
            confidence=float(value.get("confidence", 1.0)),
            authority=float(value.get("authority", 1.0)),
            source_checksum=value.get("source_checksum"),
            created_at=str(value.get("created_at", utcnowiso())),
            last_verified_at=value.get("last_verified_at"),
            metadata=dict(value.get("metadata", {})),
        )
@dataclass(frozen=True)
class ActivationEngram:
    memory_id: str
    address: SparseCode
    context_code: SparseCode
    payload: Mapping[str, Any]
    provenance: SourceRef
    version: EngramVersion
    salience: float
    timestamp: str
    @property
    def ishypothetical(self) -> bool:
        return self.version.status == "hypothetical"
