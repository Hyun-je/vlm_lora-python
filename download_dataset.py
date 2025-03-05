import os
import json
import time
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm
from functools import partial
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBomb warnings

def save_image_safely(image, path):
    """
    Safely save image without EXIF handling
    """
    try:
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        # Save without EXIF data
        image.save(path, format='JPEG', quality=95)
    except Exception as e:
        print(f"\nError converting/saving image: {str(e)}")
        return False
    return True

def create_robust_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def download_with_retry(
    dataset_name: str,
    output_dir: str,
    cache_dir: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5
):
    """
    Download dataset with retry logic
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        output_dir: Directory to save the dataset
        cache_dir: Cache directory for huggingface datasets
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    for attempt in range(max_retries):
        try:
            return load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                num_proc=4,  # Parallel processing
                verification_mode="no_checks"  # Less strict verification
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
            print(f"\nAttempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

def download_dataset(
    dataset_name: str = "lmms-lab/LLaVA-NeXT-Data",
    output_dir: str = "./data",
    cache_dir: Optional[str] = None
):
    """
    Download dataset and images for offline use with improved error handling.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        output_dir: Directory to save the dataset
        cache_dir: Cache directory for huggingface datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {dataset_name} to {output_path}")
    
    # Download dataset with retry logic
    try:
        dataset = download_with_retry(dataset_name, output_dir, cache_dir)
    except Exception as e:
        print(f"\nFatal error downloading dataset: {str(e)}")
        raise
    
    # Save dataset info
    dataset_info = {
        "name": dataset_name,
        "num_rows": len(dataset["train"]),
        "features": str(dataset["train"].features),
        "split_sizes": {split: len(data) for split, data in dataset.items()}
    }
    
    with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    # Create robust session for image downloads
    session = create_robust_session()
    
    # Save each split
    for split in dataset.keys():
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving {split} split...")
        try:
            dataset[split].save_to_disk(str(split_dir))
        except Exception as e:
            print(f"\nError saving {split} split: {str(e)}")
            continue
        
        # Download and save images if they exist
        if "image" in dataset[split].features:
            image_dir = split_dir / "images"
            image_dir.mkdir(exist_ok=True)
            
            print(f"\nDownloading images for {split} split...")
            for example in tqdm(dataset[split]):
                if example.get("image"):
                    image_path = image_dir / f"{example['id']}.jpg"
                    if not image_path.exists():
                        try:
                            # Get the PIL image
                            pil_image = example["image"]
                            # Save image without EXIF data
                            if not save_image_safely(pil_image, image_path):
                                print(f"\nSkipping image {example['id']} due to save error")
                                continue
                        except Exception as e:
                            print(f"\nError processing image {example['id']}: {str(e)}")
                            continue
    
    print(f"\nDataset successfully downloaded to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download dataset for offline use")
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/LLaVA-NeXT-Data",
                      help="Name of the dataset on HuggingFace Hub")
    parser.add_argument("--output_dir", type=str, default="./data",
                      help="Directory to save the dataset")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Cache directory for huggingface datasets")
    
    args = parser.parse_args()
    
    try:
        download_dataset(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"\nFailed to download dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main() 