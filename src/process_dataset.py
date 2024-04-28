# %%
import os
import math
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image as img
from PIL import ImageFont
from PIL.Image import Image
import matplotlib.pyplot as plt
import pickle
from transformers import LayoutLMv2Processor, LayoutLMv2ImageProcessor, LayoutLMv2PreTrainedModel, LayoutLMv2Model


from torch.utils.data import DataLoader
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from transformers.models.pix2struct.image_processing_pix2struct import ImageDraw

# %%
# SITE_ROOT = Path(__file__).parent.parent / "datasets/example-data-garaz-cz/extended_output_data/garaz"
SITE_ROOT = Path("..") / "datasets/example-seznam/seznamzpravy"
# SITE_ROOT = Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/idnes/")

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

# %%
def traverse_site_directory(root_dir, new_dir):
    data = {"image": [], "segment_boxes": [], "id": [], "html": [], "hierarchy": [], "all_boxes": [], "wrappers": []}

    if not os.path.exists(new_dir):
            os.makedirs(new_dir) 

    # print(root_dir)
    for root, dirs, _ in os.walk(root_dir, topdown=True):
        if "bounding-boxes" not in dirs:  # skip dirs, which have incomplete data, they are not usable
            continue
        else:
            image_dir = os.path.join(root, "screenshot")
            box_dir = os.path.join(root, "bounding-boxes")
            html_dir = os.path.join(root, "html")
            hierarchy_dir = os.path.join(root, "hierarchy")

            for file in os.listdir(image_dir):
                image_file = os.path.join(image_dir, file)
                box_file = os.path.join(box_dir, file.replace(".png", ".pickle"))
                html_file = os.path.join(html_dir, file.replace(".png", ".html"))
                hierarchy_file = os.path.join(hierarchy_dir, file.replace(".png", ".pickle"))

                if os.path.exists(box_file) and os.path.exists(html_file) and os.path.exists(hierarchy_file) and os.path.exists(image_file):
                      # TODO(filip): copy image file into a folder (each DS will have its own folder)
                      #              the image path will be relative to the root of the DS, so that
                      #              it is portable
                    with open(box_file, "rb") as f:
                        segments = pickle.load(f)
                        modified_segments = {}
                        filtered_segments = {}
                        wrappers = {}
                        for k,v in segments.items():
                            modified_segments[k] = [{"box": b, "class_id": c} for c,b in v.items()]
                            filtered_segments[k] = [{"box": b, "class_id": c} for c,b in v.items() if c in cls2id.keys()]
                            wrappers[k] = [{"wrapper": b} for c,b in v.items() if c == "wrapper"][0]

                        data["all_boxes"].append(modified_segments)
                        data["segment_boxes"].append(filtered_segments)
                        data["wrappers"].append(wrappers)

                    image_file_id = os.path.basename(root_dir) + "-" + str(Path(root).name) + "-" + Path(image_file).stem # [site_name]-[subdir_num]-[file_num]
                    data["id"].append(image_file_id)
                    data["html"].append(html_file)
                    data["image"].append(image_file_id)
                    
                    new_image_destination_path = os.path.join(new_dir, os.path.basename(image_file_id+".png"))
                    shutil.copy(image_file, new_image_destination_path)
                    
                    with open(hierarchy_file, "rb") as f:
                        data["hierarchy"].append(pickle.load(f))

    return data

# %%
site_list = [SITE_ROOT]
NEW_FLAT_DIRECTORY_PATH = "../datasets/flat/llmv2-flat-2023-04-29"

data = {}

for site in site_list:
    newData = traverse_site_directory(SITE_ROOT, NEW_FLAT_DIRECTORY_PATH)
    data.update({
        "all_boxes": data.get("all_boxes", []) + newData.get("all_boxes", []),
        "segment_boxes": data.get("segment_boxes", []) + newData.get("segment_boxes", []),
        "wrappers": data.get("wrappers", []) + newData.get("wrappers", []),
        "id": data.get("id", []) + newData.get("id", []),
        "html": data.get("html", []) + newData.get("html", []),
        "image": data.get("image", []) + newData.get("image", []),
        "hierarchy": data.get("hierarchy", []) + newData.get("hierarchy", [])
    })


# %%
print("Image sizes:", len(data["image"]))
print("Segment box sizes:", len(data["segment_boxes"]))
print("All box sizes:", len(data["all_boxes"]))
print("ID sizes:", len(data["id"]))
print("HTML sizes:", len(data["html"]))
print("Hierarchy sizes:", len(data["hierarchy"]))
print("id:", data["id"][0])
# print(data)

# %%
(data["wrappers"][0])

# %%
len(data["segment_boxes"][0].keys())

# %%
def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

# %%
def draw_boxes(image: Image, boxes, norm = True):
  draw = ImageDraw.Draw(image)

  width, height = image.size

  for comment_boxes in boxes.values():
      for box in comment_boxes:
        print(box["box"])
        # if norm:
        # box = unnormalize_ls_box(box, width, height)
        if box["box"] is None:
            continue
        draw.rectangle(box["box"], outline="blue", width=2)

# %%
def draw_cls_boxes(image: Image, boxes, labels, se_labels = None):
    print(se_labels)
    font = ImageFont.load_default() # type: ignore
    draw = ImageDraw.Draw(image)
    label2color = { "O": "violet", "text": "green", "author_name": "blue", "date_published": "orange" }

    width, height = image.size

    se_label = {
        0: "",
        1: " start",
        2: " end",
    }

    for i, (prediction, box) in enumerate(zip(labels, boxes)):
        se = ""
        if se_labels is not None:
            se = se_label[se_labels[i]]
            print(se)
        box = unnormalize_box(box, width, height)
        predicted_label = id2cls[prediction]
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label + se, fill=label2color[predicted_label], font=font)

# %%
# image = img.open(data["image"][0])
# draw_boxes(image, data["segment_boxes"][0])
# image.show()

# %%
# df = pd.DataFrame(data) # TODO tu nastava chyba nie su vsetky polia rovnako dlhe alebo co
# print(df)

# df.iloc[0]["image"].show() # uz nekladame priamo obrazky cize toto je nevyuzitelne
# print(df.iloc[0]["id"])
# print(df.iloc[0]["image"])
# print(df.iloc[0]["segment_boxes"])
# print(df.iloc[0]["html"])
# print(df.iloc[0]["hierarchy"])

# %%
# ds = Dataset.from_pandas(df)
# print(ds)

# %%
# TODO(filip): i don't think this works properly right now
def calculate_iou(box1, box2):
    x0_inter = max(box1[0], box2[0])
    y0_inter = max(box1[1], box2[1])
    x1_inter = min(box1[2], box2[2])
    y1_inter = min(box1[3], box2[3])
    
    if x1_inter < x0_inter or y1_inter < y0_inter:
        return 0.0
    
    intersection_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area
    
    return iou

# %%
def classify_bboxes(image, encoding, anot, wrappers):

    width, height = image.size

    ner_tags  = []
    start_end_tags = []
    
    start_idx = {}
    end_idx = {}
    for i,box in enumerate(encoding.boxes[0]):
        # print(block)
        ner_tags.append(0)
        start_end_tags.append(0) # assume we are not at start or end

        for k, comment_boxes in anot.items():
            wrapper_box = wrappers[k]["wrapper"]
            if calculate_iou(wrapper_box, unnormalize_box(box, width, height)) <= 0:
                continue

            
            for block in comment_boxes:
                # print(block)
                if block["box"] is None:
                    continue
                # TODO(filip): finish
                iou = calculate_iou(block["box"], unnormalize_box(box, width, height)) # TODO: fix iou -- don't think it works
                # print(iou)

                if iou > 0: # there is some overlap -- mark it with that label (iou doesn't seem to work properly...)
                    ner_tags[i] = cls2id[block["class_id"]]

                    if cls2id[block["class_id"]] != 0:
                        if i < start_idx.get(k, math.inf):
                            start_idx[k] = i
                        if i > end_idx.get(k, -1):
                            end_idx[k] = i

    for k, comment_boxes in anot.items():
        s = start_idx.get(k)
        e = end_idx.get(k)
        if s is None or e is None:  # TODO(filip): log this -- should not happen
            continue
        start_end_tags[s] = se_cls2id["B-Start"]
        start_end_tags[e] = se_cls2id["B-End"]

    # print(ner_tags)
    return ner_tags, start_end_tags

# %%
# processor = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
# assert(isinstance(processor, LayoutLMv2ImageProcessor))
#
# im = img.open(data["image"][0]).convert("RGB")
# encoding = processor(im, return_tensors="pt")
# ner_tags = classify_bboxes(im, encoding, data["segment_boxes"][0])
#
# draw_cls_boxes(im, encoding.boxes[0], ner_tags)
# im.show()


# %%
def check_all_same_length(annots):
     return len(annots["image"]) == len(annots["segment_boxes"]) == len(annots["id"]) == len(annots["html"]) == len(annots["hierarchy"])

# %%
def make_layoutv2_dataset(annots):
    assert(check_all_same_length(annots))
    words = []
    boxes = []
    images = []
    word_labels = []
    start_end_labels = []
    ids = []

    num_annots = len(annots["image"])

    processor = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
    assert(isinstance(processor, LayoutLMv2ImageProcessor))

    # print(annots.keys())

    # for id,val in annots.items():
    for idx in range(num_annots):
        print(f"{idx+1}/{num_annots}")

        ids.append(annots["id"][idx])

        image = img.open(annots["image"][idx]).convert("RGB")

        # TODO: could this be done for all images at one time?
        encoding = processor(image, return_tensors="pt")  # you can also add all tokenizer parameters here such as padding, truncation

        ner_tags, start_end_tags = classify_bboxes(image, encoding, annots["segment_boxes"][idx], annots["wrappers"][idx])

        words.append(encoding.words[0])
        boxes.append(encoding.boxes[0])
        images.append(annots["image"][idx])
        word_labels.append(ner_tags)
        start_end_labels.append(start_end_tags)

        # print(val["image"])

    word_label_feature = Sequence(feature=ClassLabel(num_classes=len(cls2id.values()), names=list(cls2id.keys())))
    se_label_feature = Sequence(feature=ClassLabel(num_classes=len(se_cls2id.values()), names=list(se_cls2id.keys())))

    ds = Dataset.from_pandas(pd.DataFrame({ 
    "image": images,
    "words": words,
    "boxes": boxes,
    "word_labels": word_labels,
    "start_end_labels": start_end_labels,
    "id": ids
   }), features=Features({
       "image": Value(dtype='string', id=None),
       "words": Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
       "boxes": Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
       "word_labels": word_label_feature,
       "start_end_labels": se_label_feature,
       "id": Value(dtype="string")
       }))

    return ds

# %%
ds = make_layoutv2_dataset(data)

# %%
ds.features

# %%
with open("../datasets/example-seznam/seznam_se_1.pkl", "wb") as f:
    pickle.dump(ds, f)

# %%
ds

# %%
item = ds[0]

# %%
item["image"]

# %%
im = img.open(item["image"])
draw_cls_boxes(im, item["boxes"], item["word_labels"], item["start_end_labels"])
im.show()
