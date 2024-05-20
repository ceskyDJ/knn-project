# %%
from time import time
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
import matplotlib.pyplot as plt
import utils as knn_utils
from transform_dataset import se_to_blob


import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pickle

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


def load_dataset(filename: str|Path) -> Dataset:
    with open(filename, "rb") as f:
        return pickle.load(f)

# %%
BATCH_SIZE = 20
TYPE = "start_end"  # Available types: "blob", "start_end"

# %%
CHECKPOINT_DIR = Path("./checkpoints")

# %%
timestamp = int(time())

# %%
check_point_name = f"final_se_garaz_lidovky_aha_auto_e15-2-layer-weighted-loss-batch-{BATCH_SIZE}-{TYPE}-" + str(timestamp)

# %%
(CHECKPOINT_DIR / check_point_name).mkdir(exist_ok=True, parents=True)

# %%
DS_ROOT = Path("../datasets/")
train_ds, test_ds = knn_utils.construct_dataset([
    "final-2023-05-09-[gara]-split",
    "final-2023-05-09-[lidovky]-split",
    "final-2023-05-09-[aha]-split",
    "final-2023-05-09-[auto]-split",
    "final-2023-05-09-[e15]-split",
])

# %%
if TYPE == "blob":
    train_ds = se_to_blob(train_ds)
    test_ds = se_to_blob(test_ds)

# %%
with open(CHECKPOINT_DIR / check_point_name / "train_ds.pkl", "wb") as f:
    pickle.dump(train_ds, f)
with open(CHECKPOINT_DIR / check_point_name / "test_ds.pkl", "wb") as f:
    pickle.dump(test_ds, f)

# %%
labels = train_ds.features["word_labels"].feature.names

if TYPE == "blob":
    blob_labels = train_ds.features["blob_labels"].feature.names
else:
    se_labels = train_ds.features["start_end_labels"].feature.names

id2cls = {k: v for k,v in enumerate(labels)}
cls2id = {v: k for k,v in enumerate(labels)}

if TYPE == "blob":
    blob_id2cls = {k: v for k,v in enumerate(blob_labels)}
    blob_cls2id = {v: k for k,v in enumerate(blob_labels)}
else:
    se_id2cls = {k: v for k,v in enumerate(se_labels)}
    se_cls2id = {v: k for k,v in enumerate(se_labels)}

# %%
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
assert(isinstance(processor, LayoutLMv2Processor))

features_dict = {
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=labels)),
}

if TYPE == "blob":
    features_dict['blob_labels'] = Sequence(ClassLabel(names=blob_labels))
else:
    features_dict['start_end_labels'] = Sequence(ClassLabel(names=se_labels))

features = Features(features_dict)

def preprocess_data(examples):
    image = [Image.open(DS_ROOT / path).convert("RGB") for path in examples["image"]]
    encoded_inputs = processor(image, examples["words"], boxes=examples["boxes"], word_labels=examples["word_labels"], stride=128,
                               padding="max_length", truncation=True, max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)

    offset_mapping = encoded_inputs.pop('offset_mapping')
    overflow_to_sample_mapping = encoded_inputs.pop('overflow_to_sample_mapping')

    if TYPE == "blob":
        encoded_inputs_blob = processor(image, examples["words"], boxes=examples["boxes"], word_labels=examples["blob_labels"], stride=128,
                                        padding="max_length", truncation=True, max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)
        encoded_inputs["blob_labels"] = encoded_inputs_blob["labels"]
    else:
        encoded_inputs_start_end = processor(image, examples["words"], boxes=examples["boxes"], word_labels=examples["start_end_labels"], stride=128,
                                             padding="max_length", truncation=True, max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)
        encoded_inputs["start_end_labels"] = encoded_inputs_start_end["labels"]

    return encoded_inputs

# %%
train_dataset = train_ds.map(preprocess_data, batched=True, features=features, batch_size=5, remove_columns=train_ds.column_names)
train_dataset.set_format(type="torch")

test_dataset = test_ds.map(preprocess_data, batched=True, features=features, batch_size=5, remove_columns=test_ds.column_names)
test_dataset.set_format(type="torch")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE) # type: ignore
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE) # type: ignore

# %%
import custom_llmv2_no_se


# %%
import gc
model = None
trainer = None
gc.collect()

# %%
model = custom_llmv2_no_se.LayoutLMv2ForCustomClassification.from_pretrained('microsoft/layoutlmv2-base-uncased', num_labels=len(cls2id))
model.config.id2label = id2cls
model.config.label2id = cls2id

# %%
def format_metrics(metrics):
    lines = []
    for m_type, res in metrics.items():
        if m_type == "test_loss" or m_type == "test_runtime" or m_type == "test_samples_per_second" or m_type == "test_steps_per_second":
            lines.append(f"{m_type}: {res}")
        else:
            lines.append(f"{m_type}:")

            for k, v in res.items():
                lines.append(f"    {k}: {v}")
    return lines

metric: Any = load_metric("seqeval")
return_entity_level_metrics = True

def compute_metrics(p):
    predictions, labels = p
    predictions_words = predictions[0]
    predictions_words = np.argmax(predictions_words, axis=2)

    predictions_se = predictions[1]
    predictions_se = np.argmax(predictions_se, axis=2)

    word_labels = labels[0]
    se_labels = labels[1]

    true_predictions = [
        [id2cls[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_words, word_labels)
    ]
    true_labels = [
        [id2cls[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions_words, word_labels)
    ]

    if TYPE == "blob":
        true_predictions_blob = [
            [blob_id2cls[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions_se, se_labels)
        ]
        true_labels_blob = [
            [blob_id2cls[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions_se, se_labels)
        ]
    else:
        true_predictions_se = [
            [se_id2cls[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions_se, se_labels)
        ]
        true_labels_se = [
            [se_id2cls[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions_se, se_labels)
        ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if TYPE == "blob":
        results_blob = metric.compute(predictions=true_predictions_blob, references=true_labels_blob)
    else:
        results_se = metric.compute(predictions=true_predictions_se, references=true_labels_se)
    if return_entity_level_metrics:
        final_results = {}
        final_word_results = {}
        final_se_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_word_results[f"{key}_{n}"] = v
            else:
                final_word_results[key] = value

        for key, value in results_blob.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_se_results[f"{key}_{n}"] = v
            else:
                final_se_results[key] = value
        final_results["words"] = final_word_results
        if TYPE == "blob":
            final_results["blob"] = final_se_results
        else:
            final_results["start_end"] = final_se_results
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
      return test_dataloader

# %%
logging_steps = 10
args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR / check_point_name), # dir to store checkpoints
    max_steps=1501,
    save_steps=500,
    logging_steps=logging_steps,
    warmup_ratio=0.1, # small warmup
    fp16=True, # mixed precision (less memory) -- requires CUDA
    push_to_hub=False, 
    # label_names=["labels", "start_end_labels"]
    label_names=["labels", "blob_labels"]
)

# %%
trainer = CommentTrainer(
    model=model,
    args=args,
    compute_metrics=compute_metrics,
    eval_dataset=train_dataset,

)

torch.cuda.empty_cache()


# %%
trainer.train()

# %%
predictions, labels, metrics = trainer.predict(test_dataset=test_dataset)


# %%
m_lines = format_metrics(metrics)
print("\n".join(m_lines))
with open(CHECKPOINT_DIR / check_point_name / "test_metrics.txt", "w") as f:
    f.write("\n".join(m_lines))



# %%
log_df = pd.DataFrame(trainer.state.log_history)
log_df

# %%
with open(CHECKPOINT_DIR / check_point_name / "log_df.pkl", "wb") as f:
    pickle.dump(log_df, f)


# %%
plt.plot(log_df.index * logging_steps, log_df["loss"], marker="o", linestyle="-", color="blue")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title(check_point_name)
plt.savefig(CHECKPOINT_DIR / check_point_name / "loss.png")
plt.show()
