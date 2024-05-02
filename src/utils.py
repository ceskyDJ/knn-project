import pickle
from datasets import Dataset


def load_dataset(filename: str) -> Dataset:
    with open(filename, "rb") as f:
        return pickle.load(f)
