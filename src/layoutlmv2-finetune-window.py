# %%
from typing import Any, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
from collections import defaultdict
from dataclasses import dataclass
from datasets.iterable_dataset import Iterable

from transformers import LayoutLMv2Processor, LayoutLMv2ImageProcessor, Trainer, TrainingArguments
from transformers import LayoutLMv2PreTrainedModel, LayoutLMv2Model
from transformers.utils import ModelOutput
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_metric
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split



import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pickle

# %%
cls2id = {
  "O": 0,
  "text": 1,
  "author_name": 2,
  "date_published": 3,
}

id2cls = {
    0 : "O",
    1 : "text",
    2 : "author_name",
    3 : "date_published",
}

se_label = {
    0: "",
    1: " start",
    2: " end",
}

# %%
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def unnormalize_ls_box(bbox, width, height):
     return [
         width * (bbox[0] / 100),
         height * (bbox[1] / 100),
         width * (bbox[2] / 100),
         height * (bbox[3] / 100),
     ]


# %%
def load_dataset(filename: str) -> Dataset:
    with open(filename, "rb") as f:
        return pickle.load(f)

# %%
# ds = load_dataset("../datasets/example-seznam/seznam_long_1_cls_info.pkl")
ds = load_dataset("../datasets/example-seznam/seznam_long_1.pkl")

# %%
# labels = ds.features["word_labels"].feature.names
labels = list(id2cls.values())

# %%
id2cls = {v: k for k,v in enumerate(labels)}
cls2id = {k: v for k,v in enumerate(labels)}

# %%
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
assert(isinstance(processor, LayoutLMv2Processor))

features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
    # 'labels': Sequence(feature=Value(dtype='int64')),


    # 'start_end_labels': Sequence(ClassLabel(names=labels)),
    # 'parent_rels': Sequence(Value(dtype='int64')),
})

# def preprocess_data(boxes, words, ner_tags, img_path):
def preprocess_data(examples):
  # print("ex:", (examples["words"]))
  image = [Image.open(path).convert("RGB") for path in examples["image"]]
  # image = Image.open(COCO_PATH / "images" / examples["image"]).convert("RGB")
  # words = words
  # boxes = boxes
  # word_labels = ner_tags
  
  encoded_inputs = processor(image, examples["words"], boxes=examples["boxes"], word_labels=examples["word_labels"], stride=128,
                             padding="max_length", truncation=True, max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)
  # print(encoded_inputs.keys())

  offset_mapping = encoded_inputs.pop('offset_mapping')
  overflow_to_sample_mapping = encoded_inputs.pop('overflow_to_sample_mapping')


# # TODO(filip): enable
#   encoded_inputs_start_end = processor(image, examples["words"], boxes=examples["boxes"], word_labels=examples["start_end"], stride=128,
#                              padding="max_length", truncation=True, max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)
 


  # encoded_inputs_start_end = encoded_inputs.pop('offset_mapping')
  # overflow_to_sample_mapping = encoded_inputs.pop('overflow_to_sample_mapping')

# # TODO(filip): enable
#   encoded_inputs["start_end_labels"] = encoded_inputs_start_end["labels"]

  # encoded_inputs["parent_rels"] = examples["parent_rels"]
  
  return encoded_inputs

# %%
nds = ds.train_test_split(test_size=0.2, shuffle=True)

# %%
# train_ds, test_ds = train_test_split(ds, test_size=0.2, shuffle=True)
# assert(isinstance(train_ds, Dataset))
# assert(isinstance(test_ds, Dataset))

# %%
train_dataset = nds["train"].map(preprocess_data, batched=True, features=features, batch_size=5, remove_columns=ds.column_names)
train_dataset.set_format(type="torch")
# train_dataset = train_dataset.to_iterable_dataset()

test_dataset = nds["test"].map(preprocess_data, batched=True, features=features, batch_size=5, remove_columns=ds.column_names)
test_dataset.set_format(type="torch")

# %%
train_dataloader = DataLoader(train_dataset, batch_size=1) # type: ignore

# %%
import custom_llmv2_no_se

# %%
check_point_name = "custom_llmv2_no_se_2"

# %%
import gc
model = None
trainer = None
gc.collect()

# %%
model = custom_llmv2_no_se.LayoutLMv2ForCustomClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(cls2id))
model.config.id2label = id2cls
model.config.label2id = cls2id

# %%
metric: Any = load_metric("seqeval")
return_entity_level_metrics = True

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2cls[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2cls[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

class CommentTrainer(Trainer):
    def get_train_dataloader(self):
      return train_dataloader

    def get_test_dataloader(self, test_dataset):
    #   return test_dataloader
      return train_dataloader

# %%
args = TrainingArguments(
    output_dir=check_point_name, # dir to store checkpoints
    max_steps=1000,
    warmup_ratio=0.1, # small warmup
    fp16=True, # mixed precision (less memory) -- requires CUDA
    push_to_hub=False, 
)

trainer = CommentTrainer(
    model=model,
    args=args,
    compute_metrics=compute_metrics,
)

torch.cuda.empty_cache()


# %%
trainer.train()

# %%
predictions, labels, metrics = trainer.predict(test_dataset)

# %%
print(metrics)

# %%
test_dataset
