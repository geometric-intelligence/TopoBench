"""Download helpers shared by dataset adapters."""

from urllib.parse import parse_qs, urlparse

import requests


def get_file_id_from_url(url: str) -> str:
    """Extract the file identifier from a supported Google Drive URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "id" in query_params:
        return query_params["id"][0]
    if "file/d/" in parsed_url.path:
        return parsed_url.path.split("/")[3]
    raise ValueError("The provided URL is not a valid Google Drive file URL.")


def download_file_from_drive(
    file_link: str,
    path_to_save: str,
    dataset_name: str,
    file_format: str = "tar.gz",
) -> None:
    """Download a Google Drive file into a dataset raw directory."""
    file_id = get_file_id_from_url(file_link)
    response = requests.get(f"https://drive.google.com/uc?id={file_id}")
    output_path = f"{path_to_save}/{dataset_name}.{file_format}"
    if response.status_code == 200:
        with open(output_path, "wb") as stream:
            stream.write(response.content)
        print("Download complete.")
    else:
        print("Failed to download the file.")


def download_file_from_link(
    file_link: str,
    path_to_save: str,
    dataset_name: str,
    file_format: str = "tar.gz",
) -> None:
    """Download a direct-link file into a dataset raw directory."""
    response = requests.get(file_link)
    output_path = f"{path_to_save}/{dataset_name}.{file_format}"
    if response.status_code == 200:
        with open(output_path, "wb") as stream:
            stream.write(response.content)
        print("Download complete.")
    else:
        print("Failed to download the file.")


__all__ = [
    "download_file_from_drive",
    "download_file_from_link",
    "get_file_id_from_url",
]
