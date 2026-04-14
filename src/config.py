import os
from dotenv import load_dotenv

load_dotenv()
KAGGLE_DATASET_HANDLE = "datadrivendecision/trump-tweets-2009-2025"
DIAL_URL = "https://ai-proxy.lab.epam.com"
EMBEDDING_MODEL_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHAT_MODEL = "gpt-4o"
API_VERSION = "2024-10-21"
API_KEY = os.environ["OPENAI_API_KEY"]
CHROMA_PATH = "./chroma_db_identity"
COLLECTION_NAME = "documents"
CHUNKING_STRATEGY = "identity"   # "identity" | "sliding_window" | "time_window" | "semantic"
MAX_TWEETS = 10000
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50