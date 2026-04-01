from ingestion.pdf_loader import PDFLoader, download_dataset

KAGGLE_DATASET_HANDLE = "jayakarakini/game-rules"

def main():
    print("=== RAG Pipeline ===")
    print("Downloading dataset...")
    pdf_paths  = download_dataset(KAGGLE_DATASET_HANDLE)

    loader = PDFLoader(ocr_enabled=True)
    docs = loader.load_files(pdf_paths, max_files=5)
    print(f"Loaded {len(docs)} documents.")
    for doc in docs:
        print(f"{doc.source}: {len(doc.non_empty_pages)} pages")
        print(doc.pages[0].metadata)
if __name__ == "__main__":
    main()