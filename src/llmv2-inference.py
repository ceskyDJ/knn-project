# %%
from pathlib import Path
from PIL import ImageDraw, ImageFont
from transformers import AutoModelForTokenClassification, LayoutLMv2ImageProcessor, LayoutLMv2Processor
from custom_llmv2_no_se import LayoutLMv2ForCustomClassification
from PIL import Image as img
from PIL.Image import Image
import torch
import numpy as np
import utils as knn_utils


# %%
DS_ROOT = Path("../datasets/")
dataset_name = "final-2023-05-09-[gara]-split"

dataset_path = DS_ROOT / f"{dataset_name}.pkl"
ds = knn_utils.load_dataset(dataset_path)

# %%
# labels = ds.features["word_labels"].feature.names
# # se_labels = ds.features["start_end_labels"].feature.names
# blob_labels = ds.features["blob_labels"].feature.names
#
# id2cls = {k: v for k,v in enumerate(labels)}
# cls2id = {v: k for k,v in enumerate(labels)}
#
# se_id2cls = {k: v for k,v in enumerate(se_labels)}
# se_cls2id = {v: k for k,v in enumerate(se_labels)}

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
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

# %%
CHECKPOINT_DIR = Path("./checkpoints/final_se_garaz_lidovky_aha_auto_e15-2-layer-1715368093")
# CHECKPOINT_DIR = Path("../checkpoints")
model_path = CHECKPOINT_DIR  / "checkpoint-1000"
# model_path = "./layoutlmv2-finetuned-window/checkpoint-1000"
model = LayoutLMv2ForCustomClassification.from_pretrained(model_path)

# %%
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", tesseract_config="-l ces")
assert(isinstance(processor, LayoutLMv2Processor))

# %%
# img_path = Path("../datasets/example-seznam/seznamzpravy/1/screenshot/6.png")
# img_path = Path("../datasets/custom_youtube/white_bg/data/yt_big_1.png")  # NOTE: this kinda works if training on yt
# img_path = Path("../datasets/example-seznam/seznamzpravy/11/screenshot/5.png")  # NOTE: not really
# img_path = Path("../datasets/custom_sz/sz_1.png")  # NOTE: seems to work
# img_path = Path("../datasets/custom/auto-crop-1.png")  # NOTE: nope
img_path = Path("../datasets/example-seznam/seznamzpravy/4/screenshot/4.png")   # NOTE: works ok
# img_path = Path("../datasets/example-seznam/garaz/5/screenshot/3.png")   # NOTE: kinda works
# img_path = Path("../datasets/example-seznam/garaz/5/screenshot/2.png")   # NOTE: kinda works (no se)
# img_path = Path("../datasets/example-seznam/garaz/6/screenshot/2.png")   # NOTE:
# img_path = Path("../datasets/example-seznam/sz-4_comments.png")   # NOTE: doesnt work
# img_path = Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/idnes/30/screenshot/1.png") # NOTE: meh
image = img.open(img_path).convert("RGB")

width, height = image.size

# %%
encoding = processor(
    image, return_tensors="pt", return_offsets_mapping=True, truncation=True, stride=128, padding="max_length", max_length=512, return_overflowing_tokens=True
)
print(encoding.keys())

offset_mapping = encoding.pop('offset_mapping')
overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
print(encoding.keys())
print(len(encoding.bbox[0]))
print(len(encoding.bbox))
# print(len(encoding["input_ids"][0]))
# encoding.to("cuda") # can move to GPU -- have to move both encoding and the model below

# %%
# Handle split documents
x = []
for i in range(0, len(encoding['image'])):
     x.append(encoding['image'][i])
x = torch.stack(x)
encoding['image'] = x

# %%
# model = model.to("cuda")
with torch.no_grad():
    outputs = model(**encoding)

# %%
# print(outputs.logits.shape)

predictions = outputs.logits.argmax(-1).squeeze().tolist()
# start_end_predictions = outputs.start_end_logits.argmax(-1).squeeze().tolist()
blob_predictions = outputs.blob_logits.argmax(-1).squeeze().tolist()
# rel_predictions = outputs.parent_rels_logits.argmax(-1).squeeze().tolist()
token_boxes = encoding.bbox.squeeze().tolist()

# print(predictions)
# print(token_boxes)
# print(len(token_boxes))

# %%
true_predictions =[]
true_boxes = []
true_start_end_predictions = []

if outputs.logits.shape[0] != 1:
    STRIDE_COUNT = 128
    for i, (pred, box, mapped) in enumerate(zip(predictions, token_boxes, offset_mapping)):
        is_subword = np.array(mapped.squeeze().tolist())[:,0] != 0
        if i == 0:
            true_predictions += [id2cls[pred_] for idx, pred_ in enumerate(pred) if (not is_subword[idx])]
            true_boxes += [unnormalize_box(box_, width, height) for idx, box_ in enumerate(box) if not is_subword[idx]]
            # true_start_end_predictions += [pred for idx, pred in enumerate(start_end_predictions) if not is_subword[idx]]
            true_start_end_predictions += [pred for idx, pred in enumerate(blob_predictions) if not is_subword[idx]]
        else:
            true_predictions += [id2cls[pred_] for idx, pred_ in enumerate(pred) if (not is_subword[idx])][1 + STRIDE_COUNT - sum(is_subword[:1 + STRIDE_COUNT]):]
            true_boxes += [unnormalize_box(box_, width, height) for idx, box_ in enumerate(box) if not is_subword[idx]][1 + STRIDE_COUNT - sum(is_subword[:1 + STRIDE_COUNT]):]
            # true_start_end_predictions += [pred for idx, pred in enumerate(start_end_predictions) if not is_subword[idx]][1 + STRIDE_COUNT - sum(is_subword[:1 + STRIDE_COUNT]):]
            true_start_end_predictions += [pred for idx, pred in enumerate(blob_predictions) if not is_subword[idx]][1 + STRIDE_COUNT - sum(is_subword[:1 + STRIDE_COUNT]):]


    import itertools
    true_start_end_predictions = list(itertools.chain.from_iterable(true_start_end_predictions))
else:
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0

    true_predictions = [id2cls[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_start_end_predictions = [pred for idx, pred in enumerate(blob_predictions) if not is_subword[idx]]
    # true_start_end_predictions = [pred for idx, pred in enumerate(start_end_predictions) if not is_subword[idx]]
    # true_rel_predictions = [pred for idx, pred in enumerate(rel_predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

# print(true_predictions)
print(true_start_end_predictions)
# print(true_boxes)

# %%
draw = ImageDraw.Draw(image)

font = ImageFont.load_default()

# def iob_to_label(label):
#     label = label[2:]
#     if not label:
#       return 'other'
#     return label

# label2color = {'author_name':'blue', 'text':'green', 'date_published':'orange', 'O':'violet', "parent_reference": "red"}
label2color = {'B-Author_name':'blue', 'B-Text':'green', 'B-Date_published':'orange', 'O':'violet', "B-Parent_reference": "red"}
label2color = {'B-Comm':'blue', 'I-Comm':'green', 'O':'violet'}
# se_label = {
#     0: "",
#     1: " START",
#     2: " END",
# }

se_label = {
    0: "O",
    1: "B-Comm",
    2: "I-Comm",
}
# se_label = {
#     0: "",
#     1: "",
#     2: "",
# }


i = 0
for prediction, se_prediction, box in zip(true_predictions, true_start_end_predictions, true_boxes):
    # predicted_label = prediction # iob_to_label(prediction).lower()
    # draw.rectangle(box, outline=label2color[predicted_label])
    draw.rectangle(box, outline=label2color[se_label[se_prediction]])
    # draw.text((box[0]+10, box[1]-10), text=predicted_label + se_label[se_prediction], fill=label2color[predicted_label], font=font)
    draw.text((box[0]+10, box[1]-10), text=se_label[se_prediction], fill=label2color[se_label[se_prediction]], font=font)
    # draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

    # if i == 2:
    #     break
    # i+=1

image.show()

# %%
image = img.open(img_path).convert("RGB")
processor_img = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased", tesseract_config="-l ces")
encoding = processor_img(
    image, return_tensors="pt"
)  # you can also add all tokenizer parameters here such as padding, truncation
print(encoding.keys())

# %%
print(encoding.words)
