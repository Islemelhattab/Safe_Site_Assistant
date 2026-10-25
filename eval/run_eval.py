import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from rag_chain import SafeSiteRAG

REFUSAL_STRING = "I could not find this in the safety documents."


def run_evaluation():
    engine = SafeSiteRAG(search_type="similarity")
    eval_file = Path(__file__).parent / "questions.json"

    with open(eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    correct_answers = 0
    correct_refusals = 0
    total_refusal_cases = 0
    latencies = []

    print("\n" + "=" * 60)
    print("STARTING SAFESITE ASSISTANT EVALUATION")
    print("=" * 60)

    for item in dataset:
        q_id = item["id"]
        q_text = item["question"]
        is_refusal_expected = item["expected_refusal"]

        try:
            result = engine.query(q_text)
            answer = result["answer"]
            latency = result["latency_ms"]
            latencies.append(latency)

            if is_refusal_expected:
                total_refusal_cases += 1
                if REFUSAL_STRING.lower() in answer.lower():
                    correct_refusals += 1
                    status = "PASS (Correct Refusal)"
                    correct_answers += 1
                else:
                    status = "FAIL (Hallucinated)"
            else:
                if REFUSAL_STRING.lower() not in answer.lower() and len(result["sources"]) > 0:
                    correct_answers += 1
                    status = "PASS"
                else:
                    status = "FAIL (Unwanted Refusal)"

            print(f"[{q_id:02d}] Status: {status} | Latency: {latency}ms")
            print(f"     Q: {q_text}")
            print(f"     A: {answer[:100]}...\n")

        except Exception as exc:
            print(f"[{q_id:02d}] Status: RETRY_ERROR ({str(exc)[:60]}...)")

        # Pause de 3 secondes pour respecter les quotas gratuits d'OpenRouter
        time.sleep(3.0)

    accuracy_pct = (correct_answers / len(dataset)) * 100
    refusal_rate = (correct_refusals / total_refusal_cases) * 100 if total_refusal_cases else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print("=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Test Questions: {len(dataset)}")
    print(f"Overall Accuracy:     {accuracy_pct:.1f}%")
    print(f"Correct Refusals:     {correct_refusals}/{total_refusal_cases} ({refusal_rate:.1f}%)")
    print(f"Average Latency:      {avg_latency:.1f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_evaluation()