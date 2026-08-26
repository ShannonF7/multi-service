"""Stable fact and conflict identities shared by both discovery tracks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.rag.service.claim_contract import CanonicalClaim


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_claim_key(claim: CanonicalClaim) -> str:
    return _digest(claim.identity_payload())


def conflict_scope_key(claim: CanonicalClaim) -> str:
    return _digest(claim.conflict_scope_payload())


def claim_keys(claim: CanonicalClaim) -> dict[str, str]:
    return {
        "canonical_claim_key": canonical_claim_key(claim),
        "conflict_scope_key": conflict_scope_key(claim),
    }

