"""
Evaluation dataset for the DJT Tweet RAG pipeline.

Each entry has:
  question          – the query sent to the RAG system (same path as a real user)
  expected_keywords – words/phrases that should appear in a correct answer
                      (case-insensitive substring match)
  category          – topic label used for grouping results in the UI
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    expected_keywords: list[str]
    category: str


EVAL_SET: list[EvalQuestion] = [
    # --- Foreign policy ---
    EvalQuestion(
        question="What did Trump say about NATO and defense spending?",
        expected_keywords=["NATO", "pay", "defense", "billion"],
        category="Foreign Policy",
    ),
    EvalQuestion(
        question="What did Trump tweet about China and trade?",
        expected_keywords=["China", "trade", "tariff", "deal"],
        category="Foreign Policy",
    ),
    EvalQuestion(
        question="What did Trump post about Russia or Putin?",
        expected_keywords=["Russia", "Putin"],
        category="Foreign Policy",
    ),
    EvalQuestion(
        question="What did Trump say about India?",
        expected_keywords=["India"],
        category="Foreign Policy",
    ),
    # --- Domestic politics ---
    EvalQuestion(
        question="What did Trump say about the 2020 election?",
        expected_keywords=["election", "vote", "fraud", "steal", "rigged"],
        category="Domestic Politics",
    ),
    EvalQuestion(
        question="What did Trump tweet about the border and immigration?",
        expected_keywords=["border", "wall", "immigr", "Mexico"],
        category="Domestic Politics",
    ),
    EvalQuestion(
        question="What did Trump say about the media or fake news?",
        expected_keywords=["fake", "media", "news", "press"],
        category="Domestic Politics",
    ),
    # --- Economy ---
    EvalQuestion(
        question="What did Trump post about the economy and jobs?",
        expected_keywords=["econom", "job", "unemploy", "GDP", "growth"],
        category="Economy",
    ),
    EvalQuestion(
        question="What did Trump say about the stock market?",
        expected_keywords=["stock", "market", "dow", "record"],
        category="Economy",
    ),
    # --- COVID ---
    EvalQuestion(
        question="What did Trump tweet about COVID or the pandemic?",
        expected_keywords=["covid", "virus", "pandemic", "vaccine", "china virus"],
        category="COVID-19",
    ),
]
