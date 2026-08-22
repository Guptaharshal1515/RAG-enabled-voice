import os
import sys
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.harness.demo_rag import initialize_rag_system


def evaluate_guardrails(test_file: str = "evaluation/guardrail_tests.json") -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print("           GUARDRAILS & SAFETY EVALUATION")
    print("=" * 65)

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    pipeline = initialize_rag_system()

    total = len(test_cases)
    passed_cases = 0
    results_detail = []

    category_stats = {}

    for tc in test_cases:
        category = tc["category"]
        query = tc["query"]
        expected = tc["expected_action"]

        if category not in category_stats:
            category_stats[category] = {"total": 0, "correct": 0}
        category_stats[category]["total"] += 1

        response = pipeline.run(query)
        actual = "refuse" if response.refusal else "answer"

        is_correct = (actual == expected)
        if is_correct:
            passed_cases += 1
            category_stats[category]["correct"] += 1

        results_detail.append({
            "category": category,
            "query": query[:60] + "..." if len(query) > 60 else query,
            "expected": expected,
            "actual": actual,
            "passed": is_correct,
            "reason": response.error,
            "latency_ms": response.latencies_ms.get("total", 0.0)
        })

    accuracy = (passed_cases / max(total, 1)) * 100

    print("\nGuardrail Category Breakdown:")
    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / stats["total"]) * 100
        print(f"  {cat:<28} : {stats['correct']}/{stats['total']} ({cat_acc:.1f}%)")

    print(f"\nOverall Guardrail Accuracy: {accuracy:.1f}% ({passed_cases}/{total} cases)")
    print("=" * 65 + "\n")

    os.makedirs("evaluation/results", exist_ok=True)
    out_file = "evaluation/results/guardrail_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": accuracy,
            "category_stats": category_stats,
            "details": results_detail
        }, f, indent=2)
    print(f"Results saved to '{out_file}'.")

    return category_stats


if __name__ == "__main__":
    evaluate_guardrails()
