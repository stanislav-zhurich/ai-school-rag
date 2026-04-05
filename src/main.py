from loaders.csv_loader import CSVLoader
from chunkers.identity import IdentityChunker
import os
import kagglehub

KAGGLE_DATASET_HANDLE = "datadrivendecision/trump-tweets-2009-2025"
#"jayakarakini/game-rules"

def main():
    print("=== RAG Pipeline ===")
    print("Downloading dataset...")
    file_paths = download_dataset(KAGGLE_DATASET_HANDLE)

    csv_loader = CSVLoader(processed_dir="data/processed")
    tweets = csv_loader.load(file_paths[0])

    chunker = IdentityChunker()
    chunks = chunker.chunk(tweets)
    print(f"Chunks: {chunks}")


def download_dataset(kaggle_dataset_handle: str, output_dir: str = "data/raw") -> list[str]:
    """Download a Kaggle dataset and return the paths of regular files only."""
    os.makedirs(output_dir, exist_ok=True)
    path = kagglehub.dataset_download(
        handle=kaggle_dataset_handle,
        output_dir=output_dir,
    )
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]
    return files

    
if __name__ == "__main__":
    main()
