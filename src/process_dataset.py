# %%
import os
import time
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

from preprocess_images import cut_off_excess, enhance_image

# %%
DS_ROOT = Path("/media/filip/warehouse/fit/knn/v2/datasets/")
# SITE_ROOT = Path(__file__).parent.parent / "datasets/example-data-garaz-cz/extended_output_data/garaz"
# SITE_ROOT = Path("..") / "datasets/example-seznam/seznamzpravy"
# GARAZ_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/garaz")
# SZ_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/seznamzpravy")
# NOVINKY_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/novinky")
# SPORT_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/sport")
# ZIVE_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/zive")
AHA_ROOT = Path("/media/filip/warehouse/fit/knn/v2/crawled-data-v2/extended_output_data/aha/")
AUTO_ROOT = Path("/media/filip/warehouse/fit/knn/v2/crawled-data-v2/extended_output_data/auto/")
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
    site_name = root_dir.parts[-1]
    data = {"image": [], "segment_boxes": [], "id": [], "html": [], "hierarchy": [], "all_boxes": [], "wrappers": []}

    if not os.path.exists(new_dir):
            os.makedirs(new_dir) 

    for root, dirs, _ in os.walk(root_dir, topdown=True):
        if "bounding-boxes" not in dirs:  # skip dirs, which have incomplete data, they are not usable
            continue
        else:
            image_dir = os.path.join(root, "screenshot")
            box_dir = os.path.join(root, "bounding-boxes")
            html_dir = os.path.join(root, "html")
            hierarchy_dir = os.path.join(root, "hierarchy")

            for file in os.listdir(image_dir):
                # skip all but the first page, since there are annot errors
                if file != "1.png":
                    continue
                image_file = os.path.join(image_dir, file)
                box_file = os.path.join(box_dir, file.replace(".png", ".pickle"))
                html_file = os.path.join(html_dir, file.replace(".png", ".html"))
                hierarchy_file = os.path.join(hierarchy_dir, file.replace(".png", ".pickle"))

                if os.path.exists(box_file) and os.path.exists(html_file) and os.path.exists(hierarchy_file) and os.path.exists(image_file):
                    with open(box_file, "rb") as f:
                        segments = pickle.load(f)
                        modified_segments = {}
                        filtered_segments = {}
                        wrappers = {}
                        for k,v in segments.items():
                            modified_segments[k] = [{"box": b, "class_id": c} for c,b in v.items()]
                            filtered_segments[k] = [{"box": b, "class_id": c} for c,b in v.items() if c in cls2id.keys()]
                            wrappers[k] = [{"wrapper": b} for c,b in v.items() if c == "wrapper"][0]



                    image_file_id = os.path.basename(root_dir) + "-" + str(Path(root).name) + "-" + Path(image_file).stem # [site_name]-[subdir_num]-[file_num]
                    new_image_destination_path = os.path.join(new_dir, os.path.basename(image_file_id+".png"))


                    cut_img = cut_off_excess(image_file, None, wrappers, site_name)

                    _, height = cut_img.size
                    if height > 3200:
                        continue

                    cut_img.save(new_image_destination_path)


                    im_enhance = img.open(new_image_destination_path)
                    im_enhance = enhance_image(im_enhance, contrast_f=1.7, bright_f=1, gray=False, binary=False)
                    im_enhance.save(new_image_destination_path)



                    data["all_boxes"].append(modified_segments)
                    data["segment_boxes"].append(filtered_segments)
                    data["wrappers"].append(wrappers)

                    data["id"].append(image_file_id)
                    data["html"].append(html_file)
                    data["image"].append(image_file_id + ".png")
                    




                    # shutil.copy(image_file, new_image_destination_path)
                    
                    with open(hierarchy_file, "rb") as f:
                        data["hierarchy"].append(pickle.load(f))

    return data

# %%
# dataset_name = "llmv2-flat-2023-04-30-[garaz_novinky_sport_zive]"
# dataset_name = "llmv2-v2-2023-05-08-[aha]"
dataset_name = "llmv2-v2-2023-05-08-[auto]-enhanced-no-bin"

# %%
# site_list = [SZ_ROOT, GARAZ_ROOT, NOVINKY_ROOT, SPORT_ROOT, ZIVE_ROOT]
site_list = [AUTO_ROOT]
# site_list = [GARAZ_ROOT, ZIVE_ROOT]
# NEW_FLAT_DIRECTORY_PATH = "../datasets/flat/llmv2-flat-2023-04-29-[speed]"
NEW_FLAT_DIRECTORY_PATH = DS_ROOT / dataset_name

data = {}

for site in site_list:
    newData = traverse_site_directory(site, NEW_FLAT_DIRECTORY_PATH)
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

  # width, height = image.size

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
            # print(se)
        box = unnormalize_box(box, width, height)
        predicted_label = id2cls[prediction]
        draw.rectangle(box, outline=label2color[predicted_label])  # type: ignore
        draw.text((box[0]+10, box[1]-10), text=predicted_label + se, fill=label2color[predicted_label], font=font)

# %%
def calculate_overlap(container, box2):
    """
    Get the overlap between box2 and the container in terms of percentage of box2 area.

    If box2 is completely within container, result is 1.0, if it is completely outside, result is 0.0.

    :param container list[float]: Container bbox
    :param box2 list[float]: Bbox of interest
    """
    container = offset_bbox(-8, 0, container)  # TODO(filip): 
    x0_inter = max(container[0], box2[0])
    y0_inter = max(container[1], box2[1])
    x1_inter = min(container[2], box2[2])
    y1_inter = min(container[3], box2[3])
    
    if x1_inter < x0_inter or y1_inter < y0_inter:
        return 0.0
    
    intersection_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
    
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return (intersection_area+0.0005) / (box2_area+0.0005)


# %%
def detect_incorrect_seznam():
    # TODO(filip): works with classified bboxes and size of image.
    #              If the proportion of "O" labels is too high in relation to
    #              the image size and the number of "author" labels -> incorrect ground-truth
    pass

# TODO(filip): func to cut image into multiple where test is less than 500

# TODO(filip): func to cut image into 3 randomly?

# %%
# %%
def offset_bbox(x: float, y: float, bbox):
    return [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]

# %%
site_map = {
    "aha": (-8, 0),
    "lupa": (-7, -200),
}


# %%
def classify_bboxes(image: Image, boxes, anot, wrappers):
    """
    Assign label to each bbox of a word in the given image.

    It has bad performance, since at worst, it has to calculate overlap with each comment wrapper for each word.

    TODO: if we could guarantee that wrappers (and the corresponding segments like author, text, ...) are y-sorted top to bottom,
    we could speed the calculation a little.

    :param image Image: Image data
    :param boxes list: List of word bboxes
    :param anot: Image annotation
    :param wrappers list: List of comment wrappers
    """

    width, height = image.size

    ner_tags  = np.zeros(len(boxes))
    start_end_tags = np.zeros(len(boxes))

    
    start_idx = {}
    end_idx = {}

    comments = [(k, comment_boxes, wrappers[k]["wrapper"]) for k, comment_boxes in anot.items()]

    for i,box in enumerate(boxes):
        box = unnormalize_box(box, width, height)
        to_del = []
        for j,(k,comment_boxes, wrapper_box) in enumerate(comments):
            # skip comment segments, which could no possibly contain the current word
            if calculate_overlap(wrapper_box, box) <= 0:
                if box[1] > wrapper_box[1]:
                    to_del.append(j)
                continue

            
            for block in comment_boxes:
                if block["box"] is None:
                    continue
                overlap = calculate_overlap(block["box"], box)

                if overlap > 0.65: # NOTE: modify if there incorrect labels, because of segment bbox overlap
                    ner_tags[i] = cls2id[block["class_id"]]

                    # non-foolproof way of finding start and end of comments
                    if cls2id[block["class_id"]] != 0:
                        if i < start_idx.get(k, math.inf):
                            start_idx[k] = i
                        if i > end_idx.get(k, -1):
                            end_idx[k] = i

    # finalize start-end label creation
    for k, comment_boxes in anot.items():
        s = start_idx.get(k)
        e = end_idx.get(k)
        if s is None or e is None:  # TODO(filip): log this -- should not happen
            continue
        start_end_tags[s] = se_cls2id["B-Start"]
        start_end_tags[e] = se_cls2id["B-End"]

    return ner_tags, start_end_tags


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

    processor = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased", tesseract_config="-l ces")
    assert(isinstance(processor, LayoutLMv2ImageProcessor))

    step = 10
    for idx in range(0, num_annots-step, step):
        print(f"{idx+1}/{num_annots}")

        ids.extend(annots["id"][idx:idx+step])

        image = [img.open(Path(NEW_FLAT_DIRECTORY_PATH) / f).convert("RGB") for f in annots["image"][idx:idx+step]]

        encoding = processor(image, return_tensors="pt")

        for i in range(step):
            ner_tags, start_end_tags = classify_bboxes(image[i], encoding.boxes[i], annots["segment_boxes"][idx+i], annots["wrappers"][idx+i])

            words.append(encoding.words[i])
            boxes.append(encoding.boxes[i])
            images.append(annots["image"][idx+i])
            word_labels.append(ner_tags)
            start_end_labels.append(start_end_tags)

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
with open( DS_ROOT / f"{dataset_name}.pkl", "wb") as f:
    pickle.dump(ds, f)

# %%
ds

# %%
item = ds[9]

# %%
item["boxes"]

# %%
print(item["image"])
im = img.open(Path(NEW_FLAT_DIRECTORY_PATH) / item["image"])
draw_cls_boxes(im, item["boxes"], item["word_labels"], item["start_end_labels"])
im.show()
# seznamzpravy-5476-6.png # has issues
