import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm

def download_dataset(
    dataset_name: str = "lmms-lab/LLaVA-NeXT-Data",
    output_dir: str = "./data",
    cache_dir: str = None
):
    """
    Download dataset and images for offline use.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        output_dir: Directory to save the dataset
        cache_dir: Cache directory for huggingface datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {dataset_name} to {output_path}")
    
    # Download dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Save dataset info
    dataset_info = {
        "name": dataset_name,
        "num_rows": len(dataset["train"]),
        "features": str(dataset["train"].features),
        "split_sizes": {split: len(data) for split, data in dataset.items()}
    }
    
    with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    # Save each split
    for split in dataset.keys():
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving {split} split...")
        dataset[split].save_to_disk(str(split_dir))
        
        # Download and save images if they exist
        if "image" in dataset[split].features:
            image_dir = split_dir / "images"
            image_dir.mkdir(exist_ok=True)
            
            print(f"\nDownloading images for {split} split...")
            for example in tqdm(dataset[split]):
                if example.get("image"):
                    image_path = image_dir / f"{example['id']}.jpg"
                    if not image_path.exists():
                        example["image"].save(image_path)
    
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