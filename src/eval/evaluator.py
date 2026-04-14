"""
RAGAS-based evaluator for the DJT Tweet RAG pipeline.

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
    High score = retrieval is sufficiently complete. Uses LLM as judge (LLMContextRecall).
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from eval.eval_dataset import EvalQuestion
from RAG import RAG


@dataclass
class EvalResult:
    question: str
    category: str
    answer: str
    contexts: list[str]
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    @property
    def combined_score(self) -> float:
        scores = [s for s in [self.faithfulness, self.context_precision, self.context_recall] if s is not None]
        return round(sum(scores) / len(scores), 3) if scores else 0.0


def _build_ragas_llm():
    """Create a LangChain-wrapped Azure OpenAI LLM for RAGAS."""
    from langchain_openai import AzureChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(AzureChatOpenAI(
        azure_endpoint=config.DIAL_URL,
        openai_api_key=config.API_KEY,
        api_version=config.API_VERSION,
        azure_deployment=config.CHAT_MODEL,
    ))


def evaluate(
    rag: RAG,
    eval_set: list[EvalQuestion],
    n_results: int = 5,
) -> list[EvalResult]:
    """
    Run every question in *eval_set* through the RAG pipeline, then score
    each result with RAGAS faithfulness, context_precision and context_recall.

    Parameters
    ----------
    rag        : initialised RAG instance
    eval_set   : list of EvalQuestion objects
    n_results  : number of chunks to retrieve per question

    Returns
    -------
    list[EvalResult] — one entry per question, in eval_set order
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
    )

    llm = _build_ragas_llm()

    # --- run RAG for every question and collect samples ----------------------
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

    # --- run RAGAS evaluation ------------------------------------------------
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
        result.faithfulness      = round(float(row.get("faithfulness", 0)), 3)
        result.context_precision = round(float(row.get("llm_context_precision_without_reference", 0)), 3)
        result.context_recall    = round(float(row.get("context_recall", 0)), 3)

    return results
