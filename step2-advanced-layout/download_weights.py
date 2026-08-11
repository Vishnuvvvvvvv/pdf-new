import os
import sys

def download_model():
    print("Installing downloading dependencies...")
    os.system("uv pip install huggingface_hub modelscope")
    
    print("\nDownloading PP-DocLayoutV2 (this may take a few minutes as it is a large AI model)...")
    
    # We will save the weights to a dedicated folder next to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, "weights", "PP-DocLayoutV2")
    os.makedirs(weights_dir, exist_ok=True)
    
    try:
        # Try downloading via HuggingFace first
        from huggingface_hub import snapshot_download
        print(f"Downloading from HuggingFace to: {weights_dir}")
        snapshot_download(
            repo_id="opendatalab/PDF-Extract-Kit-1.0", 
            local_dir=weights_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["models/Layout/PP-DocLayoutV2/*"]
        )
        print("\n[SUCCESS] Model weights downloaded successfully!")
        
    except Exception as e:
        print(f"\nHuggingFace download failed ({e}). Falling back to ModelScope...")
        try:
            from modelscope import snapshot_download as ms_download
            ms_download(
                "OpenDataLab/PDF-Extract-Kit-1.0", 
                local_dir=weights_dir,
                allow_patterns=["models/Layout/PP-DocLayoutV2/*"]
            )
            print("\n[SUCCESS] Model weights downloaded successfully from ModelScope!")
        except Exception as e2:
            print(f"\n[ERROR] Failed to download model weights: {e2}")
            sys.exit(1)
            
    print("\nYou can now run the layout extraction using:")
    print(f'uv run run_mineru_layout.py "..\\6838_THREE COLUMN CASHBOOK.pdf" --weights "weights\\PP-DocLayoutV2\\models\\Layout\\PP-DocLayoutV2"')

if __name__ == "__main__":
    download_model()
