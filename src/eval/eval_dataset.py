"""
Evaluation dataset for the DJT Tweet RAG pipeline.

Each entry has:
  question          – the query sent to the RAG system (same path as a real user)
  reference         – a concise expected answer used by LLMContextRecall as ground truth
  expected_keywords – words/phrases that should appear in a correct answer
                      (case-insensitive substring match)
  category          – topic label used for grouping results in the UI
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    reference: str
    expected_keywords: list[str]
    category: str


EVAL_SET: list[EvalQuestion] = [
    # --- Foreign policy ---
    EvalQuestion(
        question="What did Trump say about NATO and defense spending?",
        reference="Trump repeatedly criticized NATO allies for not meeting the 2% GDP defense spending target and demanded they pay their fair share, threatening to withdraw US support.",
        expected_keywords=["NATO", "pay", "defense", "billion"],
        category="Foreign Policy",
    ),
    EvalQuestion(
        question="What did Trump tweet about China and trade?",
        reference="Trump accused China of unfair trade practices, imposed tariffs, and pushed for a trade deal that would benefit American workers and reduce the trade deficit.",
        expected_keywords=["China", "trade", "tariff", "deal"],
        category="Foreign Policy",
    ),
    EvalQuestion(
        question="What did Trump post about Russia or Putin?",
        reference="Trump discussed his interactions with Putin and Russia, often pushing back on collusion allegations while also commenting on US-Russia diplomatic relations.",
        expected_keywords=["Russia", "Putin"],
        category="Foreign Policy",
    ),
    EvalQuestion(
        question="What did Trump say about India?",
        reference="Trump spoke positively about India and its relationship with the US, mentioning trade talks and his visits or calls with Indian leadership.",
        expected_keywords=["India"],
        category="Foreign Policy",
    ),
    # --- Domestic politics ---
    EvalQuestion(
        question="What did Trump say about the 2020 election?",
        reference="Trump repeatedly claimed the 2020 election was stolen or rigged through widespread voter fraud, urging supporters and officials to contest the results.",
        expected_keywords=["election", "vote", "fraud", "steal", "rigged"],
        category="Domestic Politics",
    ),
    EvalQuestion(
        question="What did Trump tweet about the border and immigration?",
        reference="Trump called for building a wall on the southern border, condemned illegal immigration, and criticized Democrats for obstructing border security measures.",
        expected_keywords=["border", "wall", "immigr", "Mexico"],
        category="Domestic Politics",
    ),
    EvalQuestion(
        question="What did Trump say about the media or fake news?",
        reference="Trump frequently attacked mainstream media outlets as 'fake news' and 'enemies of the people', accusing them of biased and dishonest reporting.",
        expected_keywords=["fake", "media", "news", "press"],
        category="Domestic Politics",
    ),
    # --- Economy ---
    EvalQuestion(
        question="What did Trump post about the economy and jobs?",
        reference="Trump boasted about record low unemployment, strong GDP growth, and job creation under his administration, crediting his tax cuts and deregulation policies.",
        expected_keywords=["econom", "job", "unemploy", "GDP", "growth"],
        category="Economy",
    ),
    EvalQuestion(
        question="What did Trump say about the stock market?",
        reference="Trump regularly highlighted record highs in the Dow Jones and stock market as proof of economic success under his presidency.",
        expected_keywords=["stock", "market", "dow", "record"],
        category="Economy",
    ),
    # --- COVID ---
    EvalQuestion(
        question="What did Trump tweet about COVID or the pandemic?",
        reference="Trump referred to COVID-19 as the 'China Virus', promoted treatments like hydroxychloroquine, and touted the rapid development of vaccines under Operation Warp Speed.",
        expected_keywords=["covid", "virus", "pandemic", "vaccine", "china virus"],
        category="COVID-19",
    ),
]
