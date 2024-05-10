# %%
from pathlib import Path
import pickle
from datasets import ClassLabel, Features, Sequence, Value
from datasets import Dataset, concatenate_datasets
import pandas as pd

# %%
se_id2cls = {
    0: "O",
    1: "B-Start",
    2: "B-End",
}

se_cls2id = {
    "O":       0,
    "B-Start": 1,
    "B-End":   2,
}


blob_id2cls = {
    0: "O",
    1: "B-Comm",
    2: "I-Comm",
}

blob_cls2id = {
    "O":       0,
    "B-Comm": 1,
    "I-Comm":   2,
}


cls2id = {
  "O": 0,
  "B-Text": 1,
  "B-Author_name": 2,
  "B-Date_published": 3,
  "B-Parent_reference": 4,
}
id2cls = {
    0 : "O",
    1 : "B-Text",
    2 : "B-Author_name",
    3 : "B-Date_published",
    4 : "B-Parent_reference",
}

# %%
def load_dataset(filename: str|Path) -> Dataset:
    with open(filename, "rb") as f:
        return pickle.load(f)


# %%
DS_ROOT = Path("/media/filip/warehouse/fit/knn/datasets/")

# %%
def se_to_blob(ds: Dataset, ds_root=DS_ROOT) -> Dataset:



    word_labels = ds["word_labels"],
    start_end_labels = ds["start_end_labels"]

    blob_labels = []

    for se_labels in start_end_labels:
        blobs = []

        in_flag = False
        for se in se_labels:
            if se == se_cls2id["B-Start"]:
                in_flag = True
                blobs.append(blob_cls2id["B-Comm"])
            elif se == se_cls2id["B-End"]:
                in_flag = False
                blobs.append(blob_cls2id["I-Comm"])
            elif in_flag:
                blobs.append(blob_cls2id["I-Comm"])
            else:
                blobs.append(blob_cls2id["O"])
        blob_labels.append(blobs)


    word_label_feature = Sequence(feature=ClassLabel(num_classes=len(cls2id.values()), names=list(cls2id.keys())))
    blob_label_feature = Sequence(feature=ClassLabel(num_classes=len(blob_cls2id.values()), names=list(blob_cls2id.keys())))


    ds = Dataset.from_pandas(pd.DataFrame({ 
    "image": ds["image"],
    "words": ds["words"],
    "boxes": ds["boxes"],
    "word_labels": ds["word_labels"],
    "blob_labels": blob_labels,
    "id": ds["id"]
    }), features=Features({
       "image": Value(dtype='string', id=None),
       "words": Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
       "boxes": Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
       "word_labels": word_label_feature,
       "blob_labels": blob_label_feature,
       "id": Value(dtype="string")
       }))

    return ds


# %%
# ds = load_dataset(DS_ROOT / "final-2023-05-09-[gara]-split.pkl")
# bds = se_to_blob(ds)
# bds["blob_labels"][0]

