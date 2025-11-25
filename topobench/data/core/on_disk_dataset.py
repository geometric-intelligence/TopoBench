import os
import torch
from typing import Optional, Callable
from torch_geometric.data import Dataset
from urllib.parse import urlparse
import time
import requests
import gzip
import tarfile
import zipfile
from tqdm import tqdm

class OnDiskDataset(Dataset):
    """
    Implementation of OnDiskDataset that processes and saves each sample individually to disk.
    """
    
    def __init__(
        self, 
        root: str, 
        transform: Optional[Callable] = None, 
        pre_transform: Optional[Callable] = None, 
        pre_filter: Optional[Callable] = None,
        download_url: Optional[str] = None,
        force_download: bool = False,
        **kwargs
    ):
        self._processed_files = None
        self.download_url = download_url
        self.force_download = force_download
        super().__init__(root, transform, pre_transform, pre_filter, **kwargs)
        
        # Auto-process if dataset doesn't exist
        if not self._is_processed():
            self.process()

    def download(self):
        """
        Download implementation with progress tracking and automatic extraction.
        """
        start_time = time.time()
        print(f"Downloading dataset to {self.raw_dir}...")
        os.makedirs(self.raw_dir, exist_ok=True)
        
        url = self.download_url
        filename = self._get_filename_from_url(self.download_url)
        filepath = os.path.join(self.raw_dir, filename)
        
        # Check if file already exists
        if not self.force_download and os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size > 0:
                print(f"File already exists: {filename} ({file_size} bytes)")
                return
        
        # Download the file
        print(f"Downloading {filename} from {url}")
        self._download_with_progress(url, filepath)
        
        # Extract if archive file
        if self._is_archive_file(filepath):
            print(f"Extracting {filename}...")
            self._extract_file(filepath)
        
        print(f"Successfully downloaded and processed {filename}")
        t_download = time.time() - start_time
        print(f"Download time is: {t_download:.2f} seconds")

    def _get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if not filename:
            # Fallback name if no filename in URL
            filename = f"downloaded_file_{hash(url)[:8]}"
        return filename
    
    def _is_archive_file(self, filepath: str) -> bool:
        """Check if file is an archive that should be extracted"""
        archive_extensions = {'.gz', '.tar', '.tar.gz', '.tgz', '.zip'}
        return any(filepath.endswith(ext) for ext in archive_extensions)

    def _raw_files_exist(self) -> bool:
        """Check if all required raw files exist"""
        if not self.download_url:
            # If no URL provided, check if raw_file_names exist
            raw_files = self.raw_file_names
            if isinstance(raw_files, str):
                raw_files = [raw_files]
            return all(os.path.exists(os.path.join(self.raw_dir, f)) for f in raw_files)
        
        # Check if downloaded file exists
        filename = self._get_filename_from_url(self.download_url)
        filepath = os.path.join(self.raw_dir, filename)
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0
    
    def _download_with_progress(self, url: str, filepath: str):
        """Download file with progress bar and resume capability"""
        # Head request to get file info
        headers = {}
        file_size = 0
        mode = 'wb'
        
        if os.path.exists(filepath) and not self.force_download:
            # Resume download
            file_size = os.path.getsize(filepath)
            headers = {'Range': f'bytes={file_size}-'}
            mode = 'ab'
        
        response = requests.get(url, stream=True, headers=headers, timeout=60)
        response.raise_for_status()
        
        # Handle different HTTP statuses
        if response.status_code == 206:  # Partial content
            print("Resuming download...")
        elif response.status_code == 200 and file_size > 0:
            print("Server doesn't support resume, restarting download...")
            file_size = 0
            mode = 'wb'
        
        # Get total size for progress bar
        total_size = int(response.headers.get('content-length', 0)) + file_size
        
        with open(filepath, mode) as file, tqdm(
            desc=f"Downloading {os.path.basename(filepath)}",
            total=total_size,
            initial=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    size = file.write(chunk)
                    progress_bar.update(size)

    def _extract_file(self, filepath: str):
        """Extract compressed/archive files: """
        filename = os.path.basename(filepath)
        
        try:
            if filename.endswith('.gz'):
                self._extract_gz(filepath)
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                self._extract_tar(filepath)
            elif filename.endswith('.tar'):
                self._extract_tar(filepath)
            elif filename.endswith('.zip'):
                self._extract_zip(filepath)
            else:
                print(f"Unknown archive format: {filename}")
        except Exception as e:
            print(f"Failed to extract {filename}: {e}")
            raise

    def _extract_gz(self, gz_path: str):
        """Extract .gz file"""
        output_path = gz_path[:-3]  # Remove .gz extension
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"Extracted: {os.path.basename(output_path)}")

    def _extract_tar(self, tar_path: str):
        """Extract .tar, .tar.gz, .tgz files"""
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(self.raw_dir)
            members = tar.getnames()
            print(f"Extracted {len(members)} files from {os.path.basename(tar_path)}")

    def _extract_zip(self, zip_path: str):
        """Extract .zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
            members = zip_ref.namelist()
            print(f"Extracted {len(members)} files from {os.path.basename(zip_path)}")

    def _is_processed(self) -> bool:
        """Check if dataset is already processed"""
        return len(self.processed_file_names) > 0

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        """Override this to return your raw data file names"""
        return ["data.pt"]

    @property
    def processed_file_names(self):
        """Return list of processed file names"""
        if self._processed_files is None:
            # Auto-discover processed files
            pattern = os.path.join(self.processed_dir, "sample_*.pt")
            files = [f for f in os.listdir(self.processed_dir) if f.startswith('sample_') and f.endswith('.pt')]
            self._processed_files = sorted(files)
        
        return self._processed_files if self._processed_files else []

    def len(self) -> int:
        """Return number of samples in dataset"""
        return len(self.processed_file_names)

    def get(self, idx: int):
        """Load a single sample by index"""
        if idx < 0 or idx >= self.len():
            raise IndexError(f"Index {idx} out of range for dataset with {self.len()} samples")
        
        file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(file_path)
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data

    def process(self):
        """Process raw data and save to processed format - override this"""
        raise NotImplementedError("Subclasses must implement process() method")