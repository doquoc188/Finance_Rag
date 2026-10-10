"""Deterministic verification using supplied text and standard-library values."""

from .numeric import (
    EvidenceMatch,
    NumericMention,
    NumericType,
    NumericVerification,
    VerificationStatus,
    extract_numeric_mentions,
    verify_numeric_claims,
)

__all__ = [
    "EvidenceMatch",
    "NumericMention",
    "NumericType",
    "NumericVerification",
    "VerificationStatus",
    "extract_numeric_mentions",
    "verify_numeric_claims",
]
