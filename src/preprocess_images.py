# %%
from pathlib import Path
import pickle
from PIL import Image as img
from PIL import ImageFont, ImageEnhance, ImageFilter
from PIL.Image import Image
import math

from transformers.models.pix2struct.image_processing_pix2struct import ImageDraw


# %%
AHA_ROOT= Path("/media/filip/warehouse/fit/knn/v2/crawled-data-v2/extended_output_data/aha/")
AUTO_ROOT= Path("/media/filip/warehouse/fit/knn/v2/crawled-data-v2/extended_output_data/auto/")
SZ_ROOT= Path("/media/filip/warehouse/fit/knn/merged-data/extended_output_data/seznamzpravy/")

# %%
cls2id = {
  "O": 0,
  "text": 1,
  "author_name": 2,
  "date_published": 3,
  "parent_reference": 4,
}
id2cls = {
    0 : "O",
    1 : "text",
    2 : "author_name",
    3 : "date_published",
    4 : "parent_reference",
}

# %%
def cut_image(image: Image, save_path, keep_top=True, percentage=0.5) -> Image:
    width, height = image.size

    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage should be between 0 and 1")

    split_height = int(height * percentage)

    if keep_top:
        splt = image.crop((0, 0, width, split_height))
    else:
        splt = image.crop((0, split_height, width, height))

    if save_path is not None:
        splt.save(save_path)
    # else:
    #     splt.show()

    image.close()
    return splt

site_map = { # (keep_top, percentage)
    # "aha": (True, 0.7),
    "aha": (True, 0.1),  # TODO(filip): find good value
    "auto": (True, 0.1),  # TODO(filip): find good value
    "blesk": (True, 0.1),  # TODO(filip): find good value
    "e15": (True, 0.1),  # TODO(filip): find good value
    "idnes": (True, 0.1),  # TODO(filip): find good value
    "isport": (True, 0.1),  # TODO(filip): find good value
    "lidovky": (True, 0.5),  # TODO(filip): find good value
    "lupa": (False, 0.4),  # TODO(filip): find good value
    "sme": (True, 0.4),  # TODO(filip): find good value
}

# %%
def find_bottom_most_wrapper(bboxes):
    max_y = 0
    for c, ws in bboxes.items():
        bottom_y = ws["wrapper"][3]
        max_y = max_y if bottom_y < max_y else bottom_y
    return max_y

def find_top_most_wrapper(bboxes):
    min_y = 0
    for _, ws in bboxes.items():
        bottom_y = ws["wrapper"][1]
        min_y = min_y if bottom_y > min_y else bottom_y
    return min_y


def cut_off_excess(image: Path|str|Image, save_path: Path|str|None, bboxes, site: str):
    if not isinstance(image, Image):
        image = img.open(image)
    if site not in site_map.keys():
        return image
    cut_args = site_map[site]

    _, height = image.size


    padding = 40
    if cut_args[0]:
        bottom_y = find_bottom_most_wrapper(bboxes) + padding
        bottom_perc = bottom_y / height
        cut_perc = max(bottom_perc, cut_args[1])
        cut_perc = min(cut_perc, 1)
    else:
        top_y = find_top_most_wrapper(bboxes) - padding
        top_perc = top_y / height
        cut_perc = min(top_perc, cut_args[1])
        cut_perc = max(0, cut_perc)

    splt_img = cut_image(image, save_path, cut_args[0], cut_perc)
    return splt_img


# %%
def repair_height(cut_height: float, box: list[float]):
    if box is None:
        return box
    return [box[0], box[1] - cut_height, box[2], box[3] - cut_height]

# %%
def split_at_boundary(boundary_wrapper, data): # -> ((im1, b1), (im2, b2))
    image, wrappers, word_boxes, rest_iterables = data
    width, height = image.size

    split_height = boundary_wrapper[3]

    print(split_height, height)

    top = image.crop((0, 0, width, split_height))
    bot = image.crop((0, split_height, width, height))

    wrappers_top = {k:w for k,w in wrappers.items() if w["wrapper"][1] < boundary_wrapper[3]}
    wrappers_bot = {k:{bn:repair_height(split_height, b) for bn,b in w.items()} for k,w in wrappers.items() if w["wrapper"][1] + 2.0 > boundary_wrapper[3]}

    if word_boxes is not None:
        word_boxes_top = []
        idx = 0
        for i,wb in enumerate(word_boxes):
            if wb[1] < boundary_wrapper[3]:
                idx = i
                word_boxes_top.append(wb)

        word_boxes_bottom = word_boxes[idx:]

        its_top = []
        its_bottom = []
        for it in rest_iterables:
            its_top.append(it[0:idx])
            its_bottom.append(it[idx:])
        return ((top, wrappers_top, word_boxes_top, its_top), (bot, wrappers_bot, word_boxes_bottom, its_bottom))
    else:
        dicts_top = []
        dicts_bottom = []
        wrp_top_ids = wrappers_top.keys()
        wrp_bot_ids = wrappers_bot.keys()
        for it in rest_iterables:
            dicts_top.append({k:s for k,s in it.items() if k in wrp_top_ids})
            dicts_bottom.append({k:[{"box":repair_height(split_height,b["box"]), "class_id": b["class_id"]} for b in s] for k,s in it.items() if k in wrp_bot_ids})

        return ((top, wrappers_top, None, dicts_top), (bot, wrappers_bot, None, dicts_bottom))


# %%
def find_closest_boundary(bboxes, height):
    closest = [0,0,0,0]
    for _, b in bboxes.items():
        b = b["wrapper"]
        if b[3] < height and b[3] > closest[3]:
            closest = b
        
    return closest



# %%
def split_into_parts(image: Image, wrappers, word_boxes, its: list[list], num_parts: int): # -> [(image, wrappers, bboxes)]
    _, height = image.size
    
    part_height = height / num_parts

    res = []

    rest = (image, wrappers, word_boxes, its)
    for part in range(num_parts - 1):
        boundary_wrapper = find_closest_boundary(wrappers, part_height * (part+1))

        b, rest = split_at_boundary(boundary_wrapper, rest)
        res.append(b)

    res.append(rest)
    return res

# %%
def split_at_max_size(image: Image, wrappers, its: list[dict], max_height: int): # -> [(image, wrappers, segments)]
    _, height = image.size
    
    excess_height = height - max_height
    if excess_height >= max_height:
        num_parts = math.ceil(height / max_height)
        part_height = height / num_parts
    else:
        num_parts = 2
        part_height = height / num_parts

    res = []

    boundary_wrapper = [0,0,0,0]
    rest = (image, wrappers, None, its)
    for part in range(num_parts - 1):
        ws = rest[1]
        boundary_wrapper = find_closest_boundary(ws, boundary_wrapper[3] + part_height)
        if sum(boundary_wrapper) == 0:
            boundary_wrapper[3] = part_height
        print(boundary_wrapper)

        b, rest = split_at_boundary(boundary_wrapper, rest)
        res.append(b)

    res.append(rest)
    return res


# %%
def increase_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)
    return enhanced_image

def increase_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def image_to_grayscale(image):
    return image.convert('L')

# %%
def enhance_image(image, contrast_f: float, bright_f: float, gray: bool, binary: bool):
    if bright_f != 1.0:
        image = increase_brightness(image, bright_f)
    if contrast_f != 1.0:
        image = increase_contrast(image, contrast_f)
    if gray:
        image = image_to_grayscale(image)
    if binary:
        image = image.point(lambda x: 0 if x < 178 else 255, '1')
    return image



# # %%
# image_path = SZ_ROOT / "2/screenshot/1.png"
# # image_path = AHA_ROOT / "138/screenshot/1.png"
# image = img.open(image_path)
# # img_path = AHA_ROOT / "138/screenshot/1.png"
# # image = img.open(img_path)
#
# with open(str(image_path).replace("screenshot", "bounding-boxes").replace("png", "pickle"), "rb") as f:
#     segments = pickle.load(f)
#     modified_segments = {}
#     filtered_segments = {}
#     wrappers = {}
#     for k,v in segments.items():
#         modified_segments[k] = [{"box": b, "class_id": c} for c,b in v.items()]
#         filtered_segments[k] = [{"box": b, "class_id": c} for c,b in v.items() if c in cls2id.keys()]
#         wrappers[k] = [{"wrapper": b} for c,b in v.items() if c == "wrapper"][0]
#
# # %%
# image.show()
# # %%
# res = split_at_max_size(image, wrappers, [modified_segments, filtered_segments], 1000)
#
#
# # %%
# res
#
# # %%
# res_idx = 2
# draw_boxes(res[res_idx][0], res[res_idx][1], res[res_idx][3][0]).show()
#
# # %%
# def draw_boxes(image: Image, wrappers, boxes, norm = True):
#     draw = ImageDraw.Draw(image)
#
#     for _, w in wrappers.items():
#         draw.rectangle(w["wrapper"], outline="blue", width=2)
#
#
#     for comment_boxes in boxes.values():
#         print(comment_boxes)
#         for box in comment_boxes:
#             print(box["box"])
#             if box["box"] is None:
#                 continue
#             draw.rectangle(box["box"], outline="blue", width=2)
#     return image


# # %%
# image = img.open(AUTO_ROOT / "388/screenshot/1.png")
# image = cut_image(image, None, True, 0.3)
#
# # %%
# image.show()
#
# # %%
# enh_img = image
# # enh_img = increase_brightness(enh_img, 1.1)
# enh_img = increase_contrast(enh_img, 1.3)
#
# enh_img = image_to_grayscale(enh_img)
#
#
# enh_img.show()
#
# # %%
# image = enh_img.convert("RGB")
