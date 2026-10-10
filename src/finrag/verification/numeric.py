"""Exact explicit-quantity checks with conservative lexical context alignment.

Dot denotes decimals; validated comma groups denote thousands, otherwise one/two
comma fractional digits denote decimals. No FX, tolerance, derived arithmetic,
locale inference, semantic entailment, or general hallucination detection.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import StrEnum


class VerificationStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_FOUND = "NOT_FOUND"


class NumericType(StrEnum):
    NUMBER = "NUMBER"
    PERCENTAGE = "PERCENTAGE"
    MONEY = "MONEY"


@dataclass(frozen=True, slots=True)
class NumericMention:
    raw: str
    value: Decimal
    numeric_type: NumericType
    currency: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    evidence_index: int
    sentence: str
    mention: NumericMention


@dataclass(frozen=True, slots=True)
class NumericVerification:
    claim_sentence: str
    mention: NumericMention
    status: VerificationStatus
    evidence: tuple[EvidenceMatch, ...] = ()


_CURRENCY = r"(?:VND(?!\w)|VNĐ(?!\w)|USD(?!\w)|đồng(?!\w)|\$)"
_SCALE = r"(?:thousand|million|billion|trillion|nghìn|ngàn|triệu|tỷ)"
_MENTION = re.compile(
    rf"(?<![\w.,+\-−–—$/])(?:(?P<prefix>{_CURRENCY})\s*)?"
    r"(?P<number>[+-]?\d+(?:[.,]\d+)*)"
    rf"(?:\s*(?P<scale>{_SCALE})(?!\w))?"
    rf"(?:\s*(?P<suffix>{_CURRENCY}))?"
    r"(?:\s*(?P<percent>%))?(?!\w|[.,]\d|[-–—/]\d)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE = re.compile(r"\s+(?:and|but|while|và|nhưng|còn)\s+|;\s*|,\s+(?=[^\W\d_])", re.IGNORECASE)
_YEAR = re.compile(r"(?:^|\b(?:in|for|year|fy|năm)\s+)((?:19|20)\d{2})\b", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an the in for of at to from by was were is are has had have it its our company "
    "reached reported reports recorded amounted stood equals equal year fy "
    "nam la cua dat trong voi".split()
)
_SCALES = {
    "thousand": 3,
    "nghin": 3,
    "ngan": 3,
    "million": 6,
    "trieu": 6,
    "billion": 9,
    "ty": 9,
    "trillion": 12,
}


def _fold(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
        if not unicodedata.combining(char)
    )


def _decimal_text(text: str) -> str | None:
    unsigned = text.lstrip("+-")
    if re.fullmatch(r"\d+", unsigned) or re.fullmatch(r"\d+\.\d+", unsigned):
        return text
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", unsigned):
        return text.replace(",", "")
    if re.fullmatch(r"\d+,\d{1,2}", unsigned):
        return text.replace(",", ".")
    return None


def _currency(token: str | None) -> str | None:
    if token is None:
        return None
    return "USD" if _fold(token) in {"usd", "$"} else "VND"


def extract_numeric_mentions(text: str) -> tuple[NumericMention, ...]:
    """Extract supported explicit notation in text order, preserving each raw span.

    Malformed grouping/mixed European notation, scientific notation, ranges, and
    accounting parentheses are unsupported and skipped rather than guessed.
    """
    mentions: list[NumericMention] = []
    for match in _MENTION.finditer(text):
        number = _decimal_text(match["number"])
        if number is None:
            continue
        if (
            text[max(0, match.start() - 1) : match.start()] == "("
            and text[match.end() : match.end() + 1] == ")"
        ):
            continue
        prefix, suffix = _currency(match["prefix"]), _currency(match["suffix"])
        if prefix and suffix and prefix != suffix:
            continue
        currency = prefix or suffix
        if match["percent"] and (currency or match["scale"]):
            continue
        exponent = _SCALES[_fold(match["scale"])] if match["scale"] else 0
        # Exact for long financial mantissas, independent of ambient Decimal precision.
        with localcontext() as context:
            context.prec = len(number) + exponent + 2
            value = Decimal(number) * (Decimal(10) ** exponent)
        numeric_type = (
            NumericType.PERCENTAGE
            if match["percent"]
            else NumericType.MONEY
            if currency
            else NumericType.NUMBER
        )
        mentions.append(NumericMention(match.group(), value, numeric_type, currency))
    return tuple(mentions)


@dataclass(frozen=True, slots=True)
class _ContextualMention:
    sentence: str
    mention: NumericMention
    topic: frozenset[str]
    years: frozenset[str]
    ambiguous: bool


def _contextual_mentions(text: str) -> tuple[_ContextualMention, ...]:
    claims: list[_ContextualMention] = []
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        years = frozenset(_YEAR.findall(sentence))
        for clause in _CLAUSE.split(sentence):
            mentions = extract_numeric_mentions(clause)
            topic = frozenset(re.findall(r"\w+", _fold(_MENTION.sub(" ", clause)))) - _STOPWORDS
            for mention in mentions:
                is_year = mention.raw in years
                same_type = [
                    candidate
                    for candidate in mentions
                    if candidate.numeric_type == mention.numeric_type
                    and candidate.currency == mention.currency
                    and (candidate.raw in years) == is_year
                ]
                claims.append(
                    _ContextualMention(sentence, mention, topic, years, len(same_type) > 1)
                )
    return tuple(claims)


def verify_numeric_claims(answer: str, evidence: Sequence[str]) -> tuple[NumericVerification, ...]:
    """Compare explicit values only; weak/unrelated contexts return NOT_FOUND.

    Topic tokens and explicit year anchors must match exactly after numeric masking
    and stopword removal. Conflicting compatible evidence takes precedence, retaining
    both matching and conflicting passages. Bare unequal values cannot contradict.
    """
    candidates = [
        (index, candidate)
        for index, text in enumerate(evidence)
        for candidate in _contextual_mentions(text)
    ]
    results: list[NumericVerification] = []
    for claim in _contextual_mentions(answer):
        matches: list[EvidenceMatch] = []
        conflict = False
        for index, candidate in candidates:
            if claim.ambiguous or candidate.ambiguous:
                continue
            if (
                claim.mention.numeric_type != candidate.mention.numeric_type
                or claim.mention.currency != candidate.mention.currency
                or claim.topic != candidate.topic
                or claim.years != candidate.years
                or (claim.mention.raw in claim.years) != (candidate.mention.raw in candidate.years)
            ):
                continue
            equivalent = claim.mention.value == candidate.mention.value
            if equivalent or claim.topic:
                matches.append(EvidenceMatch(index, candidate.sentence, candidate.mention))
                conflict = conflict or not equivalent
        status = (
            VerificationStatus.CONTRADICTED
            if conflict
            else VerificationStatus.SUPPORTED
            if matches
            else VerificationStatus.NOT_FOUND
        )
        results.append(NumericVerification(claim.sentence, claim.mention, status, tuple(matches)))
    return tuple(results)
