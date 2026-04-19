"""Streamlit UI package.

Modules are deliberately thin and single-purpose so ``app.py`` can stay a
short orchestrator:

* :mod:`ui.constants`      - UI-level constants (year range, platforms, thresholds)
* :mod:`ui.cache`          - ``@st.cache_*`` resource factories
* :mod:`ui.filters`        - helpers for building ChromaDB ``where`` filters
* :mod:`ui.dashboard_tab`  - Dashboard tab renderer
* :mod:`ui.rag_tab`        - RAG Assistant tab renderer
* :mod:`ui.eval_tab`       - Evaluation tab renderer
"""

from .cache import get_rag, load_tweets
from .dashboard_tab import render_dashboard_tab
from .eval_tab import render_eval_tab
from .rag_tab import render_rag_tab

__all__ = [
    "get_rag",
    "load_tweets",
    "render_dashboard_tab",
    "render_eval_tab",
    "render_rag_tab",
]
