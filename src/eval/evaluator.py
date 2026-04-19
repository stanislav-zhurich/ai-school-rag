"""RAGAS-based evaluator for the DJT Tweet RAG pipeline.

Metrics
-------
faithfulness        (float 0–1)
    Are the claims in the answer supported by the retrieved context?
    High score = no hallucination. Uses LLM as judge.

context_precision   (float 0–1)
    Are the retrieved chunks relevant to the question?
    High score = retrieval surfaces useful content. Uses LLM as judge.

context_recall      (float 0–1)
    Does the retrieved context contain enough information to answer the question?
    High score = retrieval is sufficiently complete. Uses LLM as judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import config
from eval.eval_dataset import EvalQuestion
from RAG import RAG

RAGAS_COLUMNS: dict[str, str] = {
    "faithfulness": "faithfulness",
    "context_precision": "llm_context_precision_without_reference",
    "context_recall": "context_recall",
}
"""Mapping of ``EvalResult`` field name → Ragas DataFrame column name."""


@dataclass
class EvalResult:
    """Score and trace for one evaluation question."""

    question: str
    category: str
    answer: str
    contexts: list[str]
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    @property
    def combined_score(self) -> float:
        """Unweighted mean of the three metrics, ignoring ``None`` values."""
        scores = [
            s for s in (self.faithfulness, self.context_precision, self.context_recall)
            if s is not None
        ]
        return round(sum(scores) / len(scores), 3) if scores else 0.0


def _build_ragas_llm() -> Any:
    """Create a LangChain-wrapped Azure OpenAI LLM for RAGAS."""
    from langchain_openai import AzureChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    az = config.azure()
    return LangchainLLMWrapper(AzureChatOpenAI(
        azure_endpoint=az.endpoint,
        openai_api_key=az.api_key,
        api_version=az.api_version,
        azure_deployment=az.chat_deployment,
    ))


def _pick(row: "pd.Series", column: str) -> float | None:
    """Extract a numeric score from a Ragas result row.

    Returns ``None`` when the column is missing or the value is NaN so callers
    can surface a real failure rather than silently treating it as 0.
    """
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return round(float(value), 3)


def evaluate(
    rag: RAG,
    eval_set: list[EvalQuestion],
    n_results: int = 5,
) -> list[EvalResult]:
    """Run each question through the RAG pipeline and score it with RAGAS.

    Parameters
    ----------
    rag :
        Initialised :class:`RAG` instance.
    eval_set :
        Questions to evaluate.
    n_results :
        Number of chunks to retrieve per question.

    Returns
    -------
    list[EvalResult]
        One entry per question, in ``eval_set`` order.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )

    llm = _build_ragas_llm()

    results: list[EvalResult] = []
    samples: list[SingleTurnSample] = []

    for eq in eval_set:
        answer, search_results = rag.get_answer(eq.question, n_results=n_results)
        contexts = [r.text for r in search_results]

        results.append(EvalResult(
            question=eq.question,
            category=eq.category,
            answer=answer,
            contexts=contexts,
        ))
        samples.append(SingleTurnSample(
            user_input=eq.question,
            response=answer,
            retrieved_contexts=contexts,
            reference=eq.reference,
        ))

    dataset = EvaluationDataset(samples=samples)
    ragas_result = ragas_evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm),
            LLMContextPrecisionWithoutReference(llm=llm),
            LLMContextRecall(llm=llm),
        ],
    )

    scores_df = ragas_result.to_pandas()

    for i, result in enumerate(results):
        row = scores_df.iloc[i]
        for field_name, column in RAGAS_COLUMNS.items():
            setattr(result, field_name, _pick(row, column))

    return results
