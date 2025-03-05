import os
import json
import time
from pathlib import Path
from datasets import load_dataset, disable_caching
from tqdm.auto import tqdm
from functools import partial
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import io
import numpy as np

# Disable dataset caching
disable_caching()
# Disable PIL max image size limit
Image.MAX_IMAGE_PIXELS = None

def process_and_save_image(image_data, save_path):
    """
    Process and save image data bypassing EXIF handling
    """
    try:
        # Convert to numpy array if it's a PIL Image
        if isinstance(image_data, Image.Image):
            image_array = np.array(image_data)
        else:
            image_array = image_data

        # Create new image from array
        img = Image.fromarray(image_array)
        
        # Convert to RGB if needed
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Save without any EXIF data
        img.save(save_path, format='JPEG', quality=95)
        return True
    except Exception as e:
        print(f"\nError saving image: {str(e)}")
        return False

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
    """
    for attempt in range(max_retries):
        try:
            return load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                verification_mode="no_checks"
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
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {dataset_name} to {output_path}")
    
    try:
        dataset = download_with_retry(dataset_name, output_dir, cache_dir)
    except Exception as e:
        print(f"\nFatal error downloading dataset: {str(e)}")
        raise
    
    dataset_info = {
        "name": dataset_name,
        "num_rows": len(dataset["train"]),
        "features": str(dataset["train"].features),
        "split_sizes": {split: len(data) for split, data in dataset.items()}
    }
    
    with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    session = create_robust_session()
    
    for split in dataset.keys():
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving {split} split...")
        try:
            dataset[split].save_to_disk(str(split_dir))
        except Exception as e:
            print(f"\nError saving {split} split: {str(e)}")
            continue
        
        if "image" in dataset[split].features:
            image_dir = split_dir / "images"
            image_dir.mkdir(exist_ok=True)
            
            print(f"\nDownloading images for {split} split...")
            for example in tqdm(dataset[split]):
                if example.get("image"):
                    image_path = image_dir / f"{example['id']}.jpg"
                    if not image_path.exists():
                        try:
                            # Process and save image without EXIF
                            if not process_and_save_image(example["image"], image_path):
                                print(f"\nSkipping image {example['id']}")
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