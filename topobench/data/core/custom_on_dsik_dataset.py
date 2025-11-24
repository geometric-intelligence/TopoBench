import os
import torch
import glob
from typing import List, Optional, Callable
from torch_geometric.data import Dataset, Data


class CustomOnDiskDataset(Dataset):
    """
    A custome on-disk dataset that processes and saves each sample individually to Solves memory bottlenecks of InMemoryDataset for large-scale datasets.
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self._processed_files = None
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if not self._is_processed():
            self.process()

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        """MANDATORY: Return list of raw file names"""
        return ["data.pt"]  # Override this in subclasses

    @property
    def processed_file_names(self) -> List[str]:
        """MANDATORY: Return list of processed file names"""
        # Auto-discover or generate file names
        if self._processed_files is None:
            pattern = os.path.join(self.processed_dir, "sample_*.pt")
            files = glob.glob(pattern)
            self._processed_files = [os.path.basename(f) for f in sorted(files)]
        
        return self._processed_files if self._processed_files else []

    def download(self):
        """MANDATORY: Download raw data"""
        os.makedirs(self.raw_dir, exist_ok=True)

    def process(self):
        """MANDATORY: Process raw data to processed format"""
        print("Processing dataset...")
        
        # Create processed directory
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Download if raw files don't exist
        if not self._raw_files_exist():
            self.download()
        
        # Process all samples
        sample_idx = 0
        while True:
            try:
                # Process single sample
                data = self._process_single_sample(sample_idx)
                
                if data is None:  # Stop when no more samples
                    break
                
                # Apply pre-filter
                if self.pre_filter is not None and not self.pre_filter(data):
                    sample_idx += 1
                    continue
                
                # Apply pre-transform
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                # Save individual sample
                filename = f"sample_{sample_idx:06d}.pt"
                torch.save(data, os.path.join(self.processed_dir, filename))
                
                sample_idx += 1
                
            except StopIteration:
                break  # No more samples to process
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                sample_idx += 1
                continue
        
        print(f"Processing complete: {sample_idx} samples processed")

    def _process_single_sample(self, idx: int) -> Optional[Data]:
        if idx >= 1000:  # Stop after 1000 samples in this example
            return None
            
        # Generate sample data
        num_nodes = 10 + (idx % 100)  # Varying sizes
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        y = torch.tensor([idx % 10])  # 10 classes
        
        return Data(x=x, edge_index=edge_index, y=y, idx=idx)

    def len(self) -> int:
        """MANDATORY: Return number of samples"""
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """MANDATORY: Load a single sample"""
        if idx < 0 or idx >= self.len():
            raise IndexError(f"Sample index {idx} out of range")
        
        file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(file_path)
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data

    def _is_processed(self) -> bool:
        """Check if dataset is already processed"""
        return len(self.processed_file_names) > 0

    def _raw_files_exist(self) -> bool:
        """Check if raw files exist"""
        return all(os.path.exists(os.path.join(self.raw_dir, f)) 
                  for f in self.raw_file_names)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)} samples)'