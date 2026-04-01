from ingestion.pdf_loader import CLIPEmbedder, PDFLoader, ImageExtractor
import os
import kagglehub

KAGGLE_DATASET_HANDLE = "jayakarakini/game-rules"

def main():
    print("=== RAG Pipeline ===")
    print("Downloading dataset...")
    pdf_paths  = download_dataset(KAGGLE_DATASET_HANDLE)

    embedder = CLIPEmbedder()

    # Text pipeline
    loader = PDFLoader(ocr_enabled=True)
    docs = loader.load_files(pdf_paths)

    # Image pipeline
    extractor = ImageExtractor(embedder)
    images = extractor.extract_from_files(pdf_paths, image_dir="data/processed/images")

def download_dataset(kaggle_dataset_handle: str, output_dir: str = "data/raw") -> list[str]:
    """Download a Kaggle dataset and return the paths of all PDF files found."""
    os.makedirs(output_dir, exist_ok=True)
    path = kagglehub.dataset_download(
        handle=kaggle_dataset_handle,
        output_dir=output_dir,
    )
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(".pdf")
    ]
    return files

    
if __name__ == "__main__":
    main()