"""Run RAGAS evaluation on the financial RAG pipeline."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from retrieval.rag_pipeline import FinancialRAG
from evaluation.eval_dataset import EVAL_DATASET

RESULTS_PATH = Path(__file__).parent / "results.json"


def build_ragas_dataset(rag: FinancialRAG) -> Dataset:
    """Run each eval question through the RAG pipeline and collect outputs."""
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in EVAL_DATASET:
        print(f"[eval] Querying: {item['question'][:70]}...")
        result = rag.query(
            item["question"],
            company_filter=item.get("company"),
            year_filter=item.get("year"),
        )
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([s["excerpt"] for s in result["sources"]])
        rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def main():
    """Run the full RAGAS evaluation and print/save results."""
    print("Loading RAG pipeline ...")
    rag = FinancialRAG()

    print(f"Building eval dataset from {len(EVAL_DATASET)} Q&A pairs ...")
    dataset = build_ragas_dataset(rag)

    print("Running RAGAS evaluation ...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )

    df = results.to_pandas()
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(df[["question", "faithfulness", "answer_relevancy", "context_recall"]].to_string(index=False))
    print("\nMean scores:")
    for col in ["faithfulness", "answer_relevancy", "context_recall"]:
        print(f"  {col:<25}: {df[col].mean():.4f}")

    output = {
        "mean_faithfulness": float(df["faithfulness"].mean()),
        "mean_answer_relevancy": float(df["answer_relevancy"].mean()),
        "mean_context_recall": float(df["context_recall"].mean()),
        "per_question": df.to_dict(orient="records"),
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
