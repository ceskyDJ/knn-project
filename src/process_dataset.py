# %%
from collections import defaultdict
import os
import math
from typing import Any
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image as img
from PIL import ImageFont
from PIL.Image import Image
import pickle
from transformers import LayoutLMv2ImageProcessor

from datasets import ClassLabel, Dataset, Features, Sequence, Value
from transformers.models.pix2struct.image_processing_pix2struct import ImageDraw

from preprocess_images import cut_off_excess, split_at_max_size

# %%
DS_ROOT = Path("/media/filip/warehouse/fit/knn/datasets/")
RAW_DATA_ROOT = Path("/media/filip/warehouse/fit/knn/merged-data/")

# SITE_ROOT = Path(__file__).parent.parent / "datasets/example-data-garaz-cz/extended_output_data/garaz"
# SITE_ROOT = Path("..") / "datasets/example-seznam/seznamzpravy"
GARAZ_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/garaz")
# SZ_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/seznamzpravy")
# NOVINKY_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/novinky")
# SPORT_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/sport")
# ZIVE_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/zive")
AHA_ROOT = Path("/media/filip/warehouse/fit/knn/v3/crawled-data-v3/extended_output_data/aha/")
AUTO_ROOT = Path("/media/filip/warehouse/fit/knn/v3/crawled-data-v3/extended_output_data/auto/")
LIDOVKY_ROOT = Path("/media/filip/warehouse/fit/knn/v3/crawled-data-v3/extended_output_data/lidovky/")
E15_ROOT = Path("/media/filip/warehouse/fit/knn/v3/crawled-data-v3/extended_output_data/e15/")
# SITE_ROOT = Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/idnes/")

# %%
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

se_id2cls = {
    0: "O",
    1: "B-Start",
    2: "B-End",
}

se_cls2id = {
    "O": 0,
    "B-Start": 1,
    "B-End": 2,
}

# %%
MAX_HEIGHTS = defaultdict(lambda: math.inf, {
    "garaz": 1500,
    "seznamzpravy": 1500,
    "sport": 1500,
    "aha": 1500,
    "auto": 2000,
    "lidovky": 1500,
    "e15": 3500,
})


# %%
def filter_author_in_date(segments: dict[str, list[dict[str, Any]]]) -> bool:
    for _, boxes in segments.items():
        author_bounding_box = [b for b in boxes if b["class_id"] == "author_name"][0]["box"]
        date_bounding_box = [b for b in boxes if b["class_id"] == "date_published"][0]["box"]

        if author_bounding_box[0] >= date_bounding_box[0] and author_bounding_box[1] >= date_bounding_box[1] and \
                author_bounding_box[2] <= date_bounding_box[2] and author_bounding_box[3] <= date_bounding_box[3]:
            return True
    return False


# %%
def traverse_site_directory(root_dir, new_dir, dataset_name: str):
    site_name = root_dir.parts[-1]
    data = {"image": [], "segment_boxes": [], "id": [], "html": [], "all_boxes": [], "wrappers": []}

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
                # if file != "1.png":
                #     continue
                image_file = os.path.join(image_dir, file)
                box_file = os.path.join(box_dir, file.replace(".png", ".pickle"))
                html_file = os.path.join(html_dir, file.replace(".png", ".html"))
                hierarchy_file = os.path.join(hierarchy_dir, file.replace(".png", ".pickle"))

                if os.path.exists(box_file) and os.path.exists(html_file) and os.path.exists(
                        hierarchy_file) and os.path.exists(image_file):
                    with open(box_file, "rb") as f:
                        segments = pickle.load(f)
                        modified_segments = {}
                        filtered_segments = {}
                        wrappers = {}
                        for k, v in segments.items():
                            modified_segments[k] = [{"box": b, "class_id": c} for c, b in v.items()]
                            filtered_segments[k] = [{"box": b, "class_id": c} for c, b in v.items() if
                                                    c in cls2id.keys()]
                            wrappers[k] = [{"wrapper": b} for c, b in v.items() if c == "wrapper"][0]

                    cut_img = cut_off_excess(image_file, None, wrappers, site_name)

                    _, height = cut_img.size

                    if height > MAX_HEIGHTS[site_name]:
                        try:
                            img_list = split_at_max_size(cut_img, wrappers, [modified_segments, filtered_segments], int(MAX_HEIGHTS[site_name]))
                        except:
                            print(f"split_at_max_size failed for: {str(Path(root).name)}{file}. Skipping")
                            continue
                        counter = 0
                    else:
                        img_list = [(cut_img, wrappers, None, [modified_segments, filtered_segments])]
                        counter = None

                    for idx, im in enumerate(img_list):
                        part_image = im[0]
                        wrappers = im[1]
                        modified_segments = im[3][0]
                        filtered_segments = im[3][1]

                        if counter is not None:
                            # [site_name]-[subdir_num]-[file_num]-[split_num]-[ds_name]
                            image_file_id = os.path.basename(root_dir) + "-" + str(Path(root).name) + "-" + Path(
                                image_file).stem + "-" + str(
                                counter) + "-" + dataset_name
                            new_image_destination_path = os.path.join(new_dir, os.path.basename(image_file_id + ".png"))
                            counter += 1
                        else:
                            # [site_name]-[subdir_num]-[file_num]-[ds_name]
                            image_file_id = os.path.basename(root_dir) + "-" + str(Path(root).name) + "-" + Path(
                                image_file).stem + "-" + dataset_name
                            new_image_destination_path = os.path.join(new_dir, os.path.basename(image_file_id + ".png"))

                        part_image.save(new_image_destination_path)

                        # TODO(filip): remove if not needed when date black
                        # im_enhance = img.open(new_image_destination_path)
                        # im_enhance = enhance_image(im_enhance, contrast_f=1.3, bright_f=1, gray=True, binary=False)
                        # im_enhance.save(new_image_destination_path)

                        # TODO(filip): only run for sites that require it
                        # skip incorrect annotations
                        if filter_author_in_date(filtered_segments):
                            print(f"Error: found author_in_date {image_file_id}")
                            continue

                        data["all_boxes"].append(modified_segments)
                        data["segment_boxes"].append(filtered_segments)
                        data["wrappers"].append(wrappers)

                        data["id"].append(image_file_id)
                        data["html"].append(html_file)
                        data["image"].append(f"{dataset_name}/{image_file_id}.png")

                    # with open(hierarchy_file, "rb") as f:
                    #     data["hierarchy"].append(pickle.load(f))

    return data

# %%
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


# %%
def draw_boxes(image: Image, boxes, norm=True):
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
def draw_cls_boxes(image: Image, boxes, labels, se_labels=None):
    font = ImageFont.load_default()  # type: ignore
    draw = ImageDraw.Draw(image)
    label2color = {"O": "violet", "text": "green", "author_name": "blue", "date_published": "orange",
                   "parent_reference": "red"}

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
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label + se, fill=label2color[predicted_label], font=font)


# %%
def calculate_overlap(container: list[float], box2: list[float]):
    """
    Get the overlap between box2 and the container in terms of percentage of box2 area.

    If box2 is completely within container, result is 1.0, if it is completely outside, result is 0.0.

    :param container: Container bbox
    :param box2: Bbox of interest
    """
    x0_inter = max(container[0], box2[0])
    y0_inter = max(container[1], box2[1])
    x1_inter = min(container[2], box2[2])
    y1_inter = min(container[3], box2[3])

    if x1_inter < x0_inter or y1_inter < y0_inter:
        return 0.0

    intersection_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)

    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return (intersection_area + 0.0005) / (box2_area + 0.0005)


# %%
def classify_bboxes(image: Image, boxes, anot, wrappers):
    """
    Assign label to each bbox of a word in the given image.

    It has bad performance, since at worst, it has to calculate overlap with each comment wrapper for each word.

    TODO: if we could guarantee that wrappers (and the corresponding segments like author, text, ...) are y-sorted top
          to bottom, we could speed the calculation a little.

    :param image: Image data
    :param boxes: List of word bboxes
    :param anot: Image annotation
    :param wrappers: List of comment wrappers
    """

    width, height = image.size

    ner_tags = np.zeros(len(boxes))
    start_end_tags = np.zeros(len(boxes))

    start_idx = {}
    end_idx = {}

    comments = [(k, comment_boxes, wrappers[k]["wrapper"]) for k, comment_boxes in anot.items()]

    total_author_bbox = sum([1 for _, comment_boxes in anot.items() if
                             len([1 for c in comment_boxes if c["class_id"] == "author_name"]) > 0])
    used_author_bbox = set()

    for i, box in enumerate(boxes):
        box = unnormalize_box(box, width, height)
        to_del = []
        for j, (k, comment_boxes, wrapper_box) in enumerate(comments):
            # skip comment segments, which could not possibly contain the current word
            wrapper_overlap = calculate_overlap(wrapper_box, box)
            if wrapper_overlap <= 0:
                if box[1] > wrapper_box[1]:
                    to_del.append(j)
                continue

            for block in comment_boxes:
                if block["box"] is None:
                    continue
                overlap = calculate_overlap(block["box"], box)

                # NOTE: modify if there are incorrect labels, because of segment bbox overlap author_name
                # and parent_reference always have priority, since they can be nested inside other segments
                if overlap > 0.60:
                    if (cls2id[block["class_id"]] == cls2id["author_name"]
                            or cls2id[block["class_id"]] == cls2id["parent_reference"] or ner_tags[i] == 0):
                        ner_tags[i] = cls2id[block["class_id"]]

                    if block["class_id"] == "author_name":
                        used_author_bbox.add(k)

                    # non-foolproof way of finding start and end of comments
                    if cls2id[block["class_id"]] != 0:
                        if i < start_idx.get(k, math.inf):
                            start_idx[k] = i
                        if i > end_idx.get(k, -1):
                            end_idx[k] = i
            # we should get here only once for each word (since it should be in exactly one wrapper)
            # so we can safely ignore all other wrappers, and move on to the next word
            continue

    if total_author_bbox != len(used_author_bbox):
        print(f"Found invalid annot: {len(used_author_bbox)}/{total_author_bbox}", end="")
        return None, None

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
    return len(annots["image"]) == len(annots["segment_boxes"]) == len(annots["id"]) == len(
        annots["html"])  # == len(annots["hierarchy"])


# %%
def make_layoutv2_dataset(annots):
    assert (check_all_same_length(annots))
    words = []
    boxes = []
    images = []
    word_labels = []
    start_end_labels = []
    ids = []

    num_annots = len(annots["image"])

    processor = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased", tesseract_config="-l ces")
    assert (isinstance(processor, LayoutLMv2ImageProcessor))

    step = 10
    for idx in range(0, num_annots - step, step):
        print(f"{idx + 1}/{num_annots}")

        image = [img.open(DS_ROOT / f).convert("RGB") for f in annots["image"][idx:idx + step]]
        local_ids = annots["id"][idx:idx + step]

        encoding = processor(image, return_tensors="pt")

        for i in range(step):
            ner_tags, start_end_tags = classify_bboxes(image[i], encoding.boxes[i], annots["segment_boxes"][idx + i],
                                                       annots["wrappers"][idx + i])

            if ner_tags is None:
                print(f" for {annots['id'][idx:idx + step][i]}. Skipping")
                os.remove(DS_ROOT / annots['image'][idx + i])
                print(f"Removed {annots['image'][idx + i]}")
                continue

            words.append(encoding.words[i])
            boxes.append(encoding.boxes[i])
            images.append(annots["image"][idx + i])
            word_labels.append(ner_tags)
            start_end_labels.append(start_end_tags)
            ids.append(local_ids[i])

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
        "boxes": Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1,
                          id=None),
        "word_labels": word_label_feature,
        "start_end_labels": se_label_feature,
        "id": Value(dtype="string")
    }))

    return ds


# %%
for site in site_list:
    newData = traverse_site_directory(site, NEW_FLAT_DIRECTORY_PATH, DATASET_NAME)
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
# print("Hierarchy sizes:", len(data["hierarchy"]))
print("id:", data["id"][0])
# print(data)

# %%
(data["wrappers"][0])

# %%
(data["segment_boxes"])

# %%
ds = make_layoutv2_dataset(data)

# %%
ds.features

# %%
with open(DS_ROOT / f"{DATASET_NAME}.pkl", "wb") as f:
    pickle.dump(ds, f)

# %%
ds

# %%
item = ds[32]

print(item["image"])
im = img.open(DS_ROOT / item["image"]).convert("RGB")
draw_cls_boxes(im, item["boxes"], item["word_labels"], item["start_end_labels"])
im.show()
# seznamzpravy-5476-6.png # has issues

# problematic:
# final-2023-05-09-[gara]-split/garaz-933-1-0-final-2023-05-09-[gara]-split.png
# final-2023-05-09-[gara]-split/garaz-933-1-1-final-2023-05-09-[gara]-split.png
