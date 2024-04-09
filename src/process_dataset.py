import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pickle


from torch.utils.data import DataLoader
from datasets import Dataset

SITE_ROOT = Path(__file__).parent.parent / "datasets/example-seznam/seznamzpravy"

def traverse_directory(root_dir):
    # TODO(filip): add hierarchy and html (maybe as options?)
    data = {"image": [], "segment_boxes": [], "id": []}

    for subdir, dirs, files in os.walk(root_dir):
        if os.path.basename(subdir) == "screenshot":
            for file in files:
                if file.endswith(".png"):
                    image_path = os.path.join(subdir, file)
                    box_folder = os.path.join(subdir.replace("screenshot", "bounding-boxes"))
                    box_file = os.path.join(box_folder, file.replace(".png", ".pickle"))
                    if os.path.exists(box_file):
                        data["image"].append(image_path)
                        data["image"].append(Image.open(image_path))#.convert("RGB"))

                        with open(box_file, "rb") as f:
                            data["segment_boxes"].append(pickle.load(f))
                        data["id"].append(os.path.basename(root_dir) + "-" + Path(image_path).stem + "-" + str(Path(subdir).parent.name))

    return data

data = traverse_directory(SITE_ROOT)
print(data)

df_raw = pd.DataFrame(data)
# print(df)

# df.iloc[0]["image"].show()
# print(df.iloc[0]["segment_boxes"])

# ds = Dataset.from_pandas(df)
# print(ds)

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

def classify_bboxes(img, encoding, anot):

  width, height = img.size

  ner_tags  = []
  for i,box in enumerate(encoding.boxes[0]):
    # print(block)
    ner_tags.append(0)

    for block in anot["bbox"]:
      # TODO(filip): finish
      # iou = calculate_iou(unnormalize_ls_box(block["box"], width, height), unnormalize_box(box, width, height)) # TODO: fix iou -- don't think it works
      # print(iou)

      if iou > 0: # there is some overlap -- mark it with that label (iou doesn't seem to work properly...)
        ner_tags[i] = block["class_id"]


  print(ner_tags)
  return ner_tags



def make_layoutv2_dataset(annots):
  words = []
  boxes = []
  images = []
  word_labels = []

  processor = LayoutLMv2ImageProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")

  for id,val in annots.items():
    # val["bbox"] # list
    # val["image"] # base name

    image = Image.open(COCO_PATH / "images" / val["image"]).convert("RGB")

    # TODO: could this be done for all images at one time?
    encoding = processor(image, return_tensors="pt")  # you can also add all tokenizer parameters here such as padding, truncation
    
    ner_tags = classify_bboxes(image, encoding, val)

    words.append(encoding.words[0])
    boxes.append(encoding.boxes[0])
    images.append(val["image"])
    word_labels.append(ner_tags)

    print(val["image"])




  ds = Dataset.from_pandas(pd.DataFrame({ 
    "image": images,
    "words": words,
    "boxes": boxes,
    "word_labels": word_labels,
    "id": id
  }))

  return ds

