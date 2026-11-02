"""
Script to download Job Salary Prediction dataset(s) into ml/data/raw/.
"""

import argparse
import os
import sys
import urllib.request
import pandas as pd

# Default high-quality datasets for Job Salary Prediction
DATASET_URLS = {
    "ds_salaries": {
        "url": "https://raw.githubusercontent.com/arminnorouzi/sparkml/main/Data/ds_salaries.csv",
        "filename": "ds_salaries.csv",
        "description": "Data Science Job Salaries (Job title, experience level, remote ratio, company size, salary in USD)",
    },
    "salary_data": {
        "url": "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Salary%20Data.csv",
        "filename": "salary_data.csv",
        "description": "Experience vs Salary dataset (Years of Experience, Salary)",
    },
}


def download_file(url: str, dest_path: str):
    print(f"Downloading from: {url}")
    print(f"Saving to: {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)

    print(f"Downloaded successfully: {len(data) / 1024:.2f} KB")


def verify_dataset(file_path: str):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise ValueError(f"File {file_path} is missing or empty.")

    df = pd.read_csv(file_path)
    print("\n" + "=" * 50)
    print(f"Dataset Verification: {os.path.basename(file_path)}")
    print("=" * 50)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))
    print("\nPreview:")
    print(df.head(3))
    print("=" * 50 + "\n")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download job salary prediction dataset")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_URLS.keys()) + ["all"],
        default="ds_salaries",
        help="Predefined dataset choice (default: ds_salaries)",
    )
    parser.add_argument(
        "--custom-url",
        type=str,
        default=None,
        help="Custom URL to download CSV dataset from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw"),
        help="Directory to save downloaded files",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.custom_url:
        filename = os.path.basename(args.custom_url.split("?")[0]) or "dataset.csv"
        if not filename.endswith(".csv"):
            filename += ".csv"
        dest_path = os.path.join(args.output_dir, filename)
        download_file(args.custom_url, dest_path)
        verify_dataset(dest_path)
        return

    targets = list(DATASET_URLS.keys()) if args.dataset == "all" else [args.dataset]

    for key in targets:
        info = DATASET_URLS[key]
        print(f"\nProcessing '{key}': {info['description']}")
        dest_path = os.path.join(args.output_dir, info["filename"])
        download_file(info["url"], dest_path)
        verify_dataset(dest_path)


if __name__ == "__main__":
    main()

