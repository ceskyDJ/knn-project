# %%
from pathlib import Path
from PIL import Image as img
from PIL import ImageFont, ImageEnhance, ImageFilter
from PIL.Image import Image


# %%
AHA_ROOT= Path("/media/filip/warehouse/fit/knn/v2/crawled-data-v2/extended_output_data/aha/")
AUTO_ROOT= Path("/media/filip/warehouse/fit/knn/v2/crawled-data-v2/extended_output_data/auto/")

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
    "blesk": (True, 0.7),  # TODO(filip): find good value
    "e15": (True, 0.7),  # TODO(filip): find good value
    "idnes": (True, 0.7),  # TODO(filip): find good value
    "isport": (True, 0.5),  # TODO(filip): find good value
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

# %%
def find_top_most_wrapper(bboxes):
    min_y = 0
    for _, ws in bboxes.items():
        bottom_y = ws["wrapper"][1]
        min_y = min_y if bottom_y > min_y else bottom_y
    return min_y


# %%
def cut_off_excess(image_path: Path|str, save_path: Path|str|None, bboxes, site: str):
    image = img.open(image_path)
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
# cut_off_excess(Path("/media/filip/warehouse/fit/knn/v2/datasets/llmv2-v2-2023-05-08-[aha]") / data["image"][6], "", data["wrappers"][6], "aha")

# %%
def split_at_boundary(boundary_wrapper, data): # -> ((im1, b1), (im2, b2))
    image, wrappers, word_boxes, rest_iterables = data
    width, height = image.size

    split_height = boundary_wrapper[3] + 5

    top = image.crop((0, 0, width, split_height))
    bot = image.crop((0, split_height, width, height))

    wrappers_top = {k:w for k,w in wrappers.items() if w[1] < boundary_wrapper[3]}
    wrappers_bot = {k:w for k,w in wrappers.items() if w[1] > boundary_wrapper[3]}

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

# %%
def find_closest_boundary(bboxes, height):
    closest = [0,0,0,0]
    for b in bboxes:
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
