import kagglehub

# Download latest version
path = kagglehub.dataset_download(handle="franciskarajki/epstein-documents", output_dir="data/raw")

print("Path to dataset files:", path)