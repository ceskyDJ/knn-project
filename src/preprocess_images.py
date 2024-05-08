# %%
from pathlib import Path
from PIL import Image as img
from PIL import ImageFont
from PIL.Image import Image


# %%
# AHA_ROOT= Path("/media/filip/warehouse/fit/knn/v2/crawled-data-v2/extended_output_data/aha/")

# %%
def split_image(image, save_path, keep_top=True, percentage=0.5):
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
    else:
        splt.show()

    image.close()
    return splt

site_map = { # (keep_top, percentage)
    # "aha": (True, 0.7),
    "aha": (True, 0.1),  # TODO(filip): find good value
    "auto": (True, 0.7),  # TODO(filip): find good value
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
    for c, ws in bboxes.items():
        bottom_y = ws["wrapper"][1]
        min_y = min_y if bottom_y > min_y else bottom_y
    return min_y


# %%
def cut_off_excess(image_path: Path|str, save_path: Path|str, bboxes, site: str):
    if site not in site_map.keys():
        return
    cut_args = site_map[site]

    image = img.open(image_path)
    width, height = image.size


    padding = 40
    if cut_args[0]:
        bottom_y = find_bottom_most_wrapper(bboxes) + padding
        bottom_perc = bottom_y / height
        cut_perc = max(bottom_perc, cut_args[1])
        cut_perc = min(cut_perc, 100)
    else:
        top_y = find_top_most_wrapper(bboxes) - padding
        top_perc = top_y / height
        cut_perc = min(top_perc, cut_args[1])
        cut_perc = max(0, cut_perc)

    splt_img = split_image(image, save_path, cut_args[0], cut_perc)
    splt_img.close()

# %%
# cut_off_excess(Path("/media/filip/warehouse/fit/knn/v2/datasets/llmv2-v2-2023-05-08-[aha]") / data["image"][6], "", data["wrappers"][6], "aha")


