"""Test script to verify Data object migration from old PyG format."""

import copy
import pickle
from pathlib import Path

import lmdb
from torch_geometric.data import Data

# Open the first LMDB file
lmdb_path = Path(
    "/Users/theos/Documents/code/TopoBench_contrib/datasets/graph/oc20/OC22_IS2RE/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total/train/data.0000.lmdb"
)

env = lmdb.open(
    str(lmdb_path.resolve()),
    subdir=False,
    readonly=True,
    lock=False,
    readahead=True,
    meminit=False,
    max_readers=1,
)

with env.begin() as txn:
    cursor = txn.cursor()
    cursor.first()
    key, value = cursor.item()
    old_data = pickle.loads(value)

print(f"Old data type: {type(old_data)}")
print(f"Old data __dict__ keys: {old_data.__dict__.keys()}")

# Try to migrate using the new approach
if "_store" not in old_data.__dict__ or any(
    k in old_data.__dict__ for k in ["x", "edge_index", "pos"]
):
    print("Detected old format (attributes in __dict__)")

    data_dict = {}
    for key, val in old_data.__dict__.items():
        if not key.startswith("_") and val is not None:
            data_dict[key] = val
            print(f"  {key}: {type(val)}")

    # Create new Data object
    new_data = Data(**data_dict)
    print(f"\nNew data type: {type(new_data)}")
    print(f"New data __dict__ keys: {new_data.__dict__.keys()}")

    # Try to copy
    try:
        copied_data = copy.copy(new_data)
        print("✅ Copy successful!")
        print(f"Copied data __dict__ keys: {copied_data.__dict__.keys()}")
    except Exception as e:
        print(f"❌ Copy failed: {e}")

env.close()
