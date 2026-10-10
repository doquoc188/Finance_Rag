"""Explicit financial quantities and conservative, deterministic evidence checks."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest

from finrag.verification import (
    NumericType,
    VerificationStatus,
    extract_numeric_mentions,
    verify_numeric_claims,
)


@pytest.mark.parametrize(
    ("raw", "value", "kind", "currency"),
    [
        ("1250", "1250", NumericType.NUMBER, None),
        ("1,250", "1250", NumericType.NUMBER, None),
        ("1,250.50", "1250.50", NumericType.NUMBER, None),
        ("12.5", "12.5", NumericType.NUMBER, None),
        ("12.5%", "12.5", NumericType.PERCENTAGE, None),
        ("12,5%", "12.5", NumericType.PERCENTAGE, None),
        ("2 thousand", "2000", NumericType.NUMBER, None),
        ("2 million", "2000000", NumericType.NUMBER, None),
        ("1.2 billion VND", "1200000000", NumericType.MONEY, "VND"),
        ("1 trillion USD", "1000000000000", NumericType.MONEY, "USD"),
        ("2 nghìn", "2000", NumericType.NUMBER, None),
        ("2 ngàn", "2000", NumericType.NUMBER, None),
        ("2 triệu VNĐ", "2000000", NumericType.MONEY, "VND"),
        ("2 tỷ VND", "2000000000", NumericType.MONEY, "VND"),
        ("2 tỷ đồng", "2000000000", NumericType.MONEY, "VND"),
        ("VND 50 billion", "50000000000", NumericType.MONEY, "VND"),
        ("USD 50 million", "50000000", NumericType.MONEY, "USD"),
        ("$50 million", "50000000", NumericType.MONEY, "USD"),
        ("-50 million VND", "-50000000", NumericType.MONEY, "VND"),
        ("$-50", "-50", NumericType.MONEY, "USD"),
        ("+12,5%", "12.5", NumericType.PERCENTAGE, None),
        ("50 BILLION vnd", "50000000000", NumericType.MONEY, "VND"),
    ],
)
def test_extract_supported_explicit_quantities(
    raw: str, value: str, kind: NumericType, currency: str | None
) -> None:
    (mention,) = extract_numeric_mentions(f"Value: {raw}.")
    assert mention.raw == raw
    assert mention.value == Decimal(value)
    assert isinstance(mention.value, Decimal)
    assert mention.numeric_type is kind
    assert mention.currency == currency


@pytest.mark.parametrize(
    "raw",
    [
        "1.250,50",
        "1.250.000",
        "1,23,456",
        "1e3",
        "50-55",
        "50–55",
        "50—55",
        "(50 VND)",
        "12/31/2025",
        "−50",
        "$50 VND",
    ],
)
def test_unsupported_formats_are_not_partially_normalized(raw: str) -> None:
    assert extract_numeric_mentions(raw) == ()


def test_exact_scale_equivalence_and_aliases() -> None:
    quantities = [
        extract_numeric_mentions(text)[0]
        for text in ("1.2 billion VND", "1,200 million VNĐ", "1,200,000,000 VND")
    ]
    assert len({(item.value, item.numeric_type, item.currency) for item in quantities}) == 1
    for text in ("1,200 million VNĐ", "1,200,000,000 VND"):
        assert (
            verify_numeric_claims("Revenue was 1.2 billion VND.", [f"Revenue was {text}."])[
                0
            ].status
            is VerificationStatus.SUPPORTED
        )


def test_type_and_currency_semantics_remain_distinct() -> None:
    percent, number, vnd, usd = (
        extract_numeric_mentions(raw)[0]
        for raw in ("12.5%", "12.5", "50 billion VND", "50 billion USD")
    )
    assert percent.value == number.value and percent.numeric_type != number.numeric_type
    assert vnd.value == usd.value and vnd.currency != usd.currency
    for answer, evidence in (
        ("Margin was 12.5%.", "Margin was 12.5."),
        ("Revenue was 50 billion VND.", "Revenue was 50 billion USD."),
        ("Revenue was 50 billion VND.", "Revenue was 55 billion USD."),
        ("Revenue was 50 billion VND.", "Revenue was 50 billion."),
    ):
        assert verify_numeric_claims(answer, [evidence])[0].status is VerificationStatus.NOT_FOUND


@pytest.mark.parametrize(
    ("answer", "evidence", "status"),
    [
        (
            "Revenue in 2025 was 50 billion VND.",
            "Revenue reached VND 50 billion in 2025.",
            VerificationStatus.SUPPORTED,
        ),
        (
            "Revenue in 2025 was 50 billion VND.",
            "Revenue in 2025 was 55 billion VND.",
            VerificationStatus.CONTRADICTED,
        ),
        (
            "Revenue was 50 billion VND.",
            "The company employed 55,000 people.",
            VerificationStatus.NOT_FOUND,
        ),
        (
            "Revenue was 50 billion VND.",
            "The company has 50 billion VND of assets.",
            VerificationStatus.NOT_FOUND,
        ),
        ("Revenue was 50 billion VND.", "Profit was 55 billion VND.", VerificationStatus.NOT_FOUND),
        (
            "FPT revenue was 50 billion VND.",
            "VNM revenue was 55 billion VND.",
            VerificationStatus.NOT_FOUND,
        ),
        (
            "Revenue in 2025 was 50 billion VND.",
            "Revenue in 2024 was 55 billion VND.",
            VerificationStatus.NOT_FOUND,
        ),
        (
            "Revenue in 2025 was 50 billion VND.",
            "Revenue was 50 billion VND.",
            VerificationStatus.NOT_FOUND,
        ),
        (
            "Q1 revenue was 50 billion VND.",
            "Q2 revenue was 55 billion VND.",
            VerificationStatus.NOT_FOUND,
        ),
        ("Doanh thu đạt 50 tỷ VNĐ.", "Doanh thu là 50 tỷ VND.", VerificationStatus.SUPPORTED),
        ("Margin was 12.5%.", "Margin was 12,5%.", VerificationStatus.SUPPORTED),
        ("Margin was -12.5%.", "Margin was 12.5%.", VerificationStatus.CONTRADICTED),
        ("Revenue was -50 VND.", "Revenue was 50 VND.", VerificationStatus.CONTRADICTED),
        (
            "Revenue was 50 billion VND.",
            "Revenue was not 50 billion VND.",
            VerificationStatus.NOT_FOUND,
        ),
    ],
)
def test_tri_state_requires_compatible_topic_type_currency_and_period(
    answer: str, evidence: str, status: VerificationStatus
) -> None:
    results = verify_numeric_claims(answer, [evidence])
    money_or_percent = [
        result for result in results if result.mention.numeric_type != NumericType.NUMBER
    ]
    assert len(money_or_percent) == 1
    result = money_or_percent[0]
    assert result.status is status
    assert result.claim_sentence == answer
    if status is VerificationStatus.NOT_FOUND:
        assert result.evidence == ()
    else:
        assert result.evidence[0].sentence == evidence
        assert result.evidence[0].evidence_index == 0


def test_multiple_claims_preserve_answer_order_and_independent_statuses() -> None:
    answer = "Revenue reached 50 billion VND and margin was 12.5%. Employees: 100."
    results = verify_numeric_claims(answer, ["Revenue was 50 billion VND. Margin was 13%."])
    assert [item.mention.raw for item in results] == ["50 billion VND", "12.5%", "100"]
    assert [item.status for item in results] == [
        VerificationStatus.SUPPORTED,
        VerificationStatus.CONTRADICTED,
        VerificationStatus.NOT_FOUND,
    ]


def test_two_money_metrics_do_not_support_or_contradict_each_other() -> None:
    answer = "Revenue was 50 billion VND and profit was 5 billion VND."
    results = verify_numeric_claims(
        answer, ["Revenue was 55 billion VND and profit was 50 billion VND."]
    )
    assert [item.status for item in results] == [VerificationStatus.CONTRADICTED] * 2
    assert results[0].evidence[0].mention.value == Decimal("55000000000")
    assert results[1].evidence[0].mention.value == Decimal("50000000000")


def test_ambiguous_same_type_values_without_clause_boundaries_are_not_guessed() -> None:
    results = verify_numeric_claims(
        "Revenue was 50 billion VND.", ["Revenue 50 billion VND profit 55 billion VND."]
    )
    assert results[0].status is VerificationStatus.NOT_FOUND


def test_period_mentions_do_not_conflict_with_plain_amounts() -> None:
    results = verify_numeric_claims("Revenue in 2025 was 100.", ["Revenue in 2025 was 100."])
    assert [result.status for result in results] == [VerificationStatus.SUPPORTED] * 2


def test_conflicting_evidence_remains_reviewable_and_has_conservative_precedence() -> None:
    evidence = ["Revenue was 50 billion VND.", "Revenue was 55 billion VND."]
    (result,) = verify_numeric_claims("Revenue was 50 billion VND.", evidence)
    assert result.status is VerificationStatus.CONTRADICTED
    assert [match.evidence_index for match in result.evidence] == [0, 1]
    assert [match.sentence for match in result.evidence] == evidence


def test_missing_evidence_and_bare_values_do_not_invent_contradictions() -> None:
    assert verify_numeric_claims("No numeric answer.", ["Revenue: 100."]) == ()
    assert verify_numeric_claims("Revenue: 100.", [])[0].status is VerificationStatus.NOT_FOUND
    assert verify_numeric_claims("100", ["101"])[0].status is VerificationStatus.NOT_FOUND
    assert verify_numeric_claims("100", ["100"])[0].status is VerificationStatus.SUPPORTED


def test_no_derived_arithmetic_or_numeric_tolerance() -> None:
    growth = verify_numeric_claims("Revenue grew 20%.", ["2024 revenue = 100\n2025 revenue = 120"])
    assert growth[0].status is VerificationStatus.NOT_FOUND
    rounded = verify_numeric_claims("Margin was 12.5%.", ["Margin was 12.5001%."])
    assert rounded[0].status is VerificationStatus.CONTRADICTED


def test_large_decimal_is_exact_independent_of_ambient_precision() -> None:
    raw = "123456789012345678901234567890.123 million VND"
    with localcontext() as context:
        context.prec = 2
        (mention,) = extract_numeric_mentions(raw)
    assert mention.value == Decimal("123456789012345678901234567890123000")


def test_results_are_immutable_deterministic_and_do_not_execute_text() -> None:
    answer = "Revenue was 50 VND. __import__('os').system('raise 1')"
    evidence = ["Revenue was 50 VND."]
    first = verify_numeric_claims(answer, evidence)
    assert first == verify_numeric_claims(answer, evidence)
    assert first[0].status is VerificationStatus.SUPPORTED
    assert isinstance(first, tuple) and isinstance(first[0].evidence, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(first[0], "status", VerificationStatus.NOT_FOUND)
