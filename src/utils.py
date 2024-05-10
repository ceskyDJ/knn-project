from pathlib import Path
import pickle
from datasets import Dataset, concatenate_datasets


def load_dataset(filename: str|Path) -> Dataset:
    with open(filename, "rb") as f:
        return pickle.load(f)


DS_ROOT = Path("/media/filip/warehouse/fit/knn/datasets/")

def construct_dataset(names: list[str], ds_root=DS_ROOT) -> tuple[Dataset, Dataset]:
    train_ds = []
    test_ds = []
    for n in names:
        ds = load_dataset(ds_root / f"{n}.pkl")

        nds = ds.train_test_split(test_size=0.2, shuffle=True)

        train_ds.append(nds["train"])
        test_ds.append(nds["test"])


    return concatenate_datasets(train_ds), concatenate_datasets(test_ds)
