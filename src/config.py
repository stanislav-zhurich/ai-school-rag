import os
from dotenv import load_dotenv

load_dotenv()
KAGGLE_DATASET_HANDLE = "datadrivendecision/trump-tweets-2009-2025"
DIAL_URL = "https://ai-proxy.lab.epam.com"
CHAT_MODEL = "gpt-4o"
API_VERSION = "2024-10-21"
API_KEY = os.environ["OPENAI_API_KEY"]
COLLECTION_NAME = "documents"
CHUNKING_STRATEGY = "identity"   # "identity" | "sliding_window" | "time_window" | "semantic"
MAX_TWEETS = 10000
PROCESSING_DIR = "data/processed"