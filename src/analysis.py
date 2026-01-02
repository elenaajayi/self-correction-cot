"""Analysis utilities for self-correction experiment results."""

from collections import defaultdict, Counter
from typing import Dict, List


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_by_edit_type(results: List[dict]) -> Dict[str, float]:
    grouped = defaultdict(list)
    for row in results:
        grouped[row["edit_type"]].append(row["correction_score"])
    return {k: _mean(v) for k, v in grouped.items()}


def analyze_by_position(results: List[dict]) -> Dict[str, float]:
    buckets = {"early": [], "mid": [], "late": []}
    for row in results:
        frac = row["position_fraction"]
        if frac < 0.33:
            buckets["early"].append(row["correction_score"])
        elif frac <= 0.66:
            buckets["mid"].append(row["correction_score"])
        else:
            buckets["late"].append(row["correction_score"])
    return {k: _mean(v) for k, v in buckets.items()}


def analyze_explicit_rate(results: List[dict]) -> float:
    if not results:
        return 0.0
    explicit = sum(1 for r in results if r.get("explicit_correction"))
    return explicit / len(results)


def analyze_correction_types(results: List[dict]) -> Dict[str, int]:
    counter = Counter(r["correction_type"] for r in results)
    return dict(counter)


def analyze_results(results: List[dict]) -> Dict[str, object]:
    """Aggregate key analysis metrics into a single dictionary."""
    return {
        "by_edit_type": analyze_by_edit_type(results),
        "by_position": analyze_by_position(results),
        "explicit_rate": analyze_explicit_rate(results),
        "correction_type_counts": analyze_correction_types(results),
    }


def print_summary(results: List[dict]) -> None:
    by_edit = analyze_by_edit_type(results)
    by_pos = analyze_by_position(results)
    explicit_rate = analyze_explicit_rate(results)
    type_counts = analyze_correction_types(results)

    print("=== Self-Correction Summary ===")
    print("Mean correction score by edit type:")
    for k, v in sorted(by_edit.items()):
        print(f"  {k}: {v:.3f}")

    print("Mean correction score by position bucket:")
    for k, v in sorted(by_pos.items()):
        print(f"  {k}: {v:.3f}")

    print(f"Explicit correction rate: {explicit_rate:.3f}")
    print("Correction type counts:")
    for k, v in sorted(type_counts.items()):
        print(f"  {k}: {v}")


def get_key_findings(results: List[dict]) -> List[str]:
    findings: List[str] = []
    by_edit = analyze_by_edit_type(results)
    if by_edit:
        best_edit = max(by_edit.items(), key=lambda x: x[1])
        findings.append(f"Best correction score edit type: {best_edit[0]} ({best_edit[1]:.3f})")

    by_pos = analyze_by_position(results)
    if by_pos:
        best_pos = max(by_pos.items(), key=lambda x: x[1])
        findings.append(f"Position bucket with strongest corrections: {best_pos[0]} ({best_pos[1]:.3f})")

    explicit_rate = analyze_explicit_rate(results)
    findings.append(f"Explicit correction rate: {explicit_rate:.3f}")

    type_counts = analyze_correction_types(results)
    if type_counts:
        dominant_type = max(type_counts.items(), key=lambda x: x[1])
        findings.append(f"Most common correction type: {dominant_type[0]} ({dominant_type[1]})")

    # Ensure 3-5 findings; pad if needed.
    if len(findings) < 3:
        findings.append("Corrections show mixed strength across edits and positions.")
    if len(findings) < 4 and explicit_rate < 0.2:
        findings.append("Explicit self-corrections are relatively rare.")
    if len(findings) < 5 and explicit_rate >= 0.2:
        findings.append("Notable proportion of explicit self-corrections observed.")

    return findings[:5]

