import os
import requests
import zipfile
from urllib.parse import urlparse

from util.constants import DATASET_OUTPUT_PATH


def download_from_zenodo(url, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    # Get filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    file_path = os.path.join(target_folder, filename)
    file_path = os.path.normpath(file_path)

    # Skip download if file exists
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
    else:
        print(f"Downloading {filename} from Zenodo...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Download complete: {file_path}")

    return file_path


def unzip_file(zip_path, extract_to, check_already_unzipped=True):
    zip_path = os.path.normpath(zip_path)
    extract_to = os.path.normpath(extract_to)

    # Check if already unzipped by checking if folder exists and isn't empty
    if check_already_unzipped and os.path.isdir(extract_to) and os.listdir(extract_to):
        print(f"Already unzipped to: {extract_to}")
        return

    print(f"Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzipping complete.")


if __name__ == "__main__":
    target_folder = DATASET_OUTPUT_PATH

    zenodo_file_urls = [
        "https://zenodo.org/records/16414473/files/csv_header_specification.pdf",
        "https://zenodo.org/records/16414473/files/overview_table.csv",
        "https://zenodo.org/records/16414473/files/data_records_csv.zip",
        "https://zenodo.org/records/16414473/files/experiments.pkl",
        "https://zenodo.org/records/16414473/files/voxel_maps.pkl",
        "https://zenodo.org/records/16414473/files/supplementary_material.zip",
    ]

    for zenodo_file_url in zenodo_file_urls:
        zip_file_path = download_from_zenodo(zenodo_file_url, target_folder)
        unzip_file(zip_file_path, target_folder, check_already_unzipped=False)
        print()

