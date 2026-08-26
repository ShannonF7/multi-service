"""Shared, provenance-free representation of a semantic claim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

GRAPH_CLAIM_TYPES = {"PROPERTY", "RELATION"}
BACKGROUND_CLAIM_TYPE = "BACKGROUND"


@dataclass(frozen=True)
class CanonicalClaim:
    """The fact identity shared by completion and open discovery."""

    domain_id: str
    subject_ref: str
    claim_type: str
    canonical_predicate: str
    normalized_value: Optional[str] = None
    object_ref: Optional[str] = None
    value_type: Optional[str] = None
    temporal_role: Optional[str] = None
    qualifiers: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        claim_type = str(self.claim_type or "").strip().upper()
        if claim_type not in GRAPH_CLAIM_TYPES | {BACKGROUND_CLAIM_TYPE}:
            raise ValueError("claim_type must be PROPERTY, RELATION, or BACKGROUND")
        if not str(self.domain_id or "").strip() or not str(self.subject_ref or "").strip():
            raise ValueError("domain_id and subject_ref are required")
        if not str(self.canonical_predicate or "").strip():
            raise ValueError("canonical_predicate is required")
        if claim_type == "PROPERTY" and not str(self.normalized_value or "").strip():
            raise ValueError("PROPERTY claims require normalized_value")
        if claim_type == "RELATION" and not str(self.object_ref or "").strip():
            raise ValueError("RELATION claims require object_ref")
        object.__setattr__(self, "claim_type", claim_type)

    def identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "domain_id": str(self.domain_id),
            "subject_ref": str(self.subject_ref),
            "claim_type": self.claim_type,
            "canonical_predicate": str(self.canonical_predicate),
            "temporal_role": str(self.temporal_role or ""),
            "qualifiers": self.qualifiers or {},
        }
        if self.claim_type == "RELATION":
            payload["object_ref"] = str(self.object_ref)
        elif self.claim_type == "PROPERTY":
            payload["normalized_value"] = str(self.normalized_value)
            payload["value_type"] = str(self.value_type or "")
        else:
            payload["normalized_value"] = str(self.normalized_value or "")
        return payload

    def conflict_scope_payload(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload.pop("normalized_value", None)
        payload.pop("object_ref", None)
        return payload

