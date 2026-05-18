"""Run RAGAS evaluation on the financial RAG pipeline."""

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from retrieval.rag_pipeline import FinancialRAG
from evaluation.eval_dataset import EVAL_DATASET

RESULTS_PATH = Path(__file__).parent / "results.json"


def build_rows(rag: FinancialRAG) -> list[dict]:
    """Run each eval question through the RAG pipeline and collect outputs."""
    rows = []
    for item in EVAL_DATASET:
        print(f"[eval] Querying: {item['question'][:70]}...")
        result = rag.query(
            item["question"],
            company_filter=item.get("company"),
            year_filter=item.get("year"),
        )
        rows.append(
            {
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": [s["excerpt"] for s in result["sources"]],
                "reference": item["ground_truth"],
            }
        )
    return rows


def main():
    """Run the full RAGAS evaluation and print/save results."""
    print("Loading RAG pipeline ...")
    rag = FinancialRAG()

    print(f"Building eval dataset from {len(EVAL_DATASET)} Q&A pairs ...")
    rows = build_rows(rag)

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory(
        "openai", model="text-embedding-3-small", client=client
    )

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_recall = ContextRecall(llm=llm)

    print("Running RAGAS evaluation ...")
    per_question = []
    for i, row in enumerate(rows, 1):
        print(f"[eval] Scoring {i}/{len(rows)} ...")
        scores = {
            "faithfulness": faithfulness.score(
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=row["retrieved_contexts"],
            ).value,
            "answer_relevancy": answer_relevancy.score(
                user_input=row["user_input"],
                response=row["response"],
            ).value,
            "context_recall": context_recall.score(
                user_input=row["user_input"],
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["reference"],
            ).value,
        }
        per_question.append(
            {"user_input": row["user_input"], **{k: float(v) for k, v in scores.items()}}
        )

    metric_names = ["faithfulness", "answer_relevancy", "context_recall"]
    means = {
        name: sum(q[name] for q in per_question) / len(per_question)
        for name in metric_names
    }

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    for q in per_question:
        print(f"\nQ: {q['user_input'][:70]}")
        for name in metric_names:
            print(f"  {name:<20}: {q[name]:.4f}")

    print("\nMean scores:")
    for name in metric_names:
        print(f"  {name:<20}: {means[name]:.4f}")

    output = {**means, "per_question": per_question}
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
