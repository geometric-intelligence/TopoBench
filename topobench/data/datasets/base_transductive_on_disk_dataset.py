import os
import os.path as osp
import shutil
import urllib.request
import zipfile
from typing import Any

import torch
from torch_geometric.data import Data, OnDiskDataset, SQLiteDatabase


class TransductiveOnDiskDataset(OnDiskDataset):
    """
    Generic loader for an already-built `OnDiskDataset`.

    Layout
    ------
    root/
      <name>/
        raw/
        processed/
          sqlite.db
          schema.pt
          handle.pt
          <memmap_dir>/*.npy

    Parameters
    ----------
    root : str
        Root directory where the dataset should be stored.
    name : str
        Name of the dataset subdirectory under `root`.
    url : str
        URL of the zipped processed data.
    backend : str, optional
        Name of the database backend, by default "sqlite".
    transform : callable, optional
        Optional transform applied on each data example.
    """

    def __init__(
        self,
        root: str,
        name: str,
        url: str,
        backend: str = "sqlite",
        transform=None,
    ):
        self.name = name
        self.url = url
        self._table = "ClusterOnDisk"

        # Triggers raw/processed checks, `download()` and `process()` as needed.
        super().__init__(root=root, transform=transform)

        schema_path = osp.join(self.processed_dir, "schema.pt")
        handle_path = osp.join(self.processed_dir, "handle.pt")

        schema = torch.load(schema_path)
        handle = torch.load(handle_path)

        self.handle = handle
        self._db = None
        self.backend = backend
        self.schema = schema

    @property
    def raw_file_names(self):
        """
        List of expected raw file names.

        Returns
        -------
        list of str
            Filenames to check in the raw directory.
        """
        # Derives the filename from the URL (e.g., "processed.zip")
        return [self.url.split("/")[-1]]

    @property
    def processed_file_names(self):
        """
        List of files that must exist to skip processing.

        Returns
        -------
        list of str
            Filenames required in the processed directory.
        """
        return ["sqlite.db", "schema.pt", "handle.pt"]

    @property
    def raw_dir(self) -> str:
        """
        Raw data directory.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """
        Processed data directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return osp.join(self.root, self.name, "processed")

    def download(self):
        """
        Download the zipped processed data.

        Raises
        ------
        ValueError
            If no URL is provided.
        """
        if self.url is None:
            raise ValueError("URL is not provided. Cannot download the dataset.")

        zip_path = self.raw_paths[0]

        print(f"Downloading {self.url} to {zip_path}...")
        urllib.request.urlretrieve(self.url, zip_path)
        print("Download complete.")

    def process(self):
        """
        Unzip downloaded data and rewrite handle paths for the new root.

        Notes
        -----
        - Unzips the archive into `processed_dir`.
        - Detects and strips a leading `processed/` prefix if present.
        - Loads `handle.pt`, rewrites `root`, `processed_dir`,
          `memmap_dir`, and all `paths` entries to match the current
          directory structure, and saves the updated handle.
        """
        zip_path = self.raw_paths[0]
        print(f"Unzipping {zip_path} to {self.processed_dir}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            names = zip_ref.namelist()

            # Detect 'processed/' prefix pattern
            has_processed_prefix = all(
                n.startswith("processed/") or n.endswith("/")
                for n in names
            )

            for member in zip_ref.infolist():
                orig_name = member.filename

                # Skip root directories
                if orig_name.endswith("/"):
                    continue

                # Optionally strip leading 'processed/' if present
                name = orig_name
                if has_processed_prefix and name.startswith("processed/"):
                    name = name[len("processed/") :]

                # If stripping makes it empty, skip
                if not name:
                    continue

                # Final destination path
                dest_path = osp.join(self.processed_dir, name)

                # Ensure parent directory exists
                os.makedirs(osp.dirname(dest_path), exist_ok=True)

                # Extract file
                with zip_ref.open(orig_name) as source, open(dest_path, "wb") as target:
                    shutil.copyfileobj(source, target)

        print("Unzipping complete.")

        # Rewrite handle.pt to be portable to the current environment
        handle_path = osp.join(self.processed_dir, "handle.pt")
        if osp.exists(handle_path):
            handle = torch.load(handle_path)

            # Old memmap directory
            old_mm_dir = handle.get("memmap_dir", None)
            if old_mm_dir is not None:
                mm_basename = osp.basename(old_mm_dir)
            else:
                # Fallback if not present; assumes "memmap" subdir
                mm_basename = "memmap"

            new_mm_dir = osp.join(self.processed_dir, mm_basename)

            old_paths = handle.get("paths", {})
            new_paths = {}
            for key, old_path in old_paths.items():
                filename = osp.basename(old_path)
                new_paths[key] = osp.join(new_mm_dir, filename)

            handle["root"] = self.root
            handle["processed_dir"] = self.processed_dir
            handle["memmap_dir"] = new_mm_dir
            handle["paths"] = new_paths

            # Ensure memmap directory exists (in case it wasn't created)
            os.makedirs(new_mm_dir, exist_ok=True)

            torch.save(handle, handle_path)
            print("Handle updated and saved.")
        else:
            print("Warning: handle.pt not found; skipping handle rewrite.")
        
    @property
    def db(self):
        """
        Database backend instance.

        Returns
        -------
        Any
            Initialized database object.

        Notes
        -----
        Lazily opens the database and caches it in `_db`.
        """
        if self._db is not None:
            return self._db

        cls = self.BACKENDS[self.backend]
        kwargs: dict[str, Any] = {}

        path = osp.join(self.processed_dir, "sqlite.db")

        if issubclass(cls, SQLiteDatabase):
            kwargs["name"] = self._table

        self._db = cls(path=path, schema=self.schema, **kwargs)
        self._numel = len(self._db)
        return self._db

    def deserialize(self, row: dict[str, Any]) -> Data:
        """
        Deserialize a database row into a `Data` object.

        Parameters
        ----------
        row : dict of str to Any
            Row returned by the backend.

        Returns
        -------
        Data
            Constructed PyG `Data` object.
        """
        data = Data()
        for key, val in row.items():
            if val is None:
                continue
            if key == "num_nodes":
                if isinstance(val, int):
                    data.num_nodes = val
                elif hasattr(val, "item"):
                    data.num_nodes = int(val.item())
                else:
                    data.num_nodes = int(val)
            else:
                setattr(data, key, val)
        return data
