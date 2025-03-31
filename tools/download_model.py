import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(
    model_name: str = "lmms-lab/llama3-llava-next-8b",
    output_dir: str = "./models",
    cache_dir: str = None,
):
    """
    Download model and tokenizer for offline use.
    
    Args:
        model_name: Name or path of the model on HuggingFace Hub
        output_dir: Directory to save the model
        cache_dir: Cache directory for huggingface
    """
    output_path = Path(output_dir) / model_name.split("/")[-1]
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {model_name} to {output_path}")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(output_path),
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "*.h5"],
        )
        print(f"\nSuccessfully downloaded to {output_path}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download model for offline use")
    parser.add_argument("--model_name", type=str, default="lmms-lab/llama3-llava-next-8b",
                      help="Name of the model on HuggingFace Hub")
    parser.add_argument("--output_dir", type=str, default="./models",
                      help="Directory to save the model")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Cache directory for huggingface")
    
    args = parser.parse_args()
    
    try:
        download_model(
            model_name=args.model_name,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"\nFailed to download model: {str(e)}")
        raise

if __name__ == "__main__":
    main() 