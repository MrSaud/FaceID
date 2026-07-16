"""Shared risk scoring for liveness responses."""

from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_thresholds(
    allow_risk: float | None,
    review_risk: float | None,
) -> tuple[float, float]:
    allow = _clamp01(float(allow_risk) if allow_risk is not None else 0.30)
    review = _clamp01(float(review_risk) if review_risk is not None else 0.60)
    if review <= allow:
        review = min(1.0, allow + 0.15)
    return allow, review


def _decision_from_risk(risk_score: float, allow: float, review: float) -> str:
    if risk_score < allow:
        return "allow"
    if risk_score < review:
        return "review"
    return "deny"


def build_risk_response(
    *,
    signals: dict[str, float | None],
    weights: dict[str, float] | None = None,
    allow_risk: float | None = None,
    review_risk: float | None = None,
    hard_fail: bool = False,
) -> dict[str, Any]:
    """
    Convert per-signal live scores (0..1, higher = more live) into:
    risk_score, confidence, decision, and rounded signal breakdown.
    """
    allow_threshold, review_threshold = _normalize_thresholds(allow_risk, review_risk)
    default_weights = weights or {}

    live_parts: list[float] = []
    weight_parts: list[float] = []
    signal_scores: dict[str, float] = {}

    for name, value in signals.items():
        if value is None:
            continue
        score = _clamp01(value)
        signal_scores[name] = round(score, 4)
        weight = float(default_weights.get(name, 1.0))
        live_parts.append(score * weight)
        weight_parts.append(weight)

    if live_parts:
        confidence = sum(live_parts) / sum(weight_parts)
    else:
        confidence = 0.0

    risk_score = 1.0 - confidence
    if hard_fail:
        risk_score = max(risk_score, review_threshold + 0.05)

    risk_score = _clamp01(risk_score)
    confidence = _clamp01(1.0 - risk_score)
    decision = _decision_from_risk(risk_score, allow_threshold, review_threshold)

    return {
        "risk_score": round(risk_score, 4),
        "confidence": round(confidence, 4),
        "decision": decision,
        "label": "live" if decision == "allow" else ("review" if decision == "review" else "spoof"),
        "is_live": decision == "allow",
        "live_score": round(confidence, 4),
        "spoof_score": round(risk_score, 4),
        "thresholds": {
            "allow": round(allow_threshold, 4),
            "review": round(review_threshold, 4),
        },
        "signals": signal_scores,
    }
