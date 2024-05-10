# Script for fixing positioning of bounding boxes counted by script running in headless mode
#
# Usage: python fix-bounding-boxes-positioning.py [path-to-extended_output_data]
import glob
import pickle
import re
import sys
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

OFFSETS = defaultdict(lambda: (0, 0), {
    'aha': (-7, 0),
    'auto': (-7, 0),
    'avmania': (-7, 0),
    'blesk': (-7, 0),
    'connect': (-7, 0),
    'doupe': (-7, 0),
    'e15': (-6, 0),
    'idnes': (-6, 0),
    'isport': (-6, 0),
    'lidovky': (-6, 0),
    'lupa': (-6, 0),
    'mobilmania': (-6, 0),
    'pravda': (6, 0),
    'sme': (-6, 0),
    'vtm': (-7, 0),
    'zive': (-6, 0)
})


def apply_offset(offset: Tuple[float, float], bounding_box: List[float]):
    x = offset[0]
    y = offset[1]

    return [bounding_box[0] + x, bounding_box[1] + y, bounding_box[2] + x, bounding_box[3] + y]


def apply_higher_level_fixes(website_name: str, bounding_boxes_value: Dict[str, Optional[List[float]]])\
        -> Dict[str, Optional[List[float]]]:
    date_bounding_box = bounding_boxes_value['date_published']
    author_bounding_box = bounding_boxes_value['author_name']

    # Zive and child servers:
    if website_name in ["zive", "doupe", "vtm", "mobilmania", "connect", "avmania"]:
        if (author_bounding_box[0] >= date_bounding_box[0] and author_bounding_box[1] >= date_bounding_box[1]
                and author_bounding_box[2] <= date_bounding_box[2] and author_bounding_box[3] <= date_bounding_box[3]):
            # Fix published date bounding box, when there is author name bounding box in it
            date_bounding_box[0] = author_bounding_box[2]
        else:
            # Fix published date bounding box, when it is full-width sized, so part of the profile picture is in there
            date_bounding_box[0] = author_bounding_box[0]

    # Pravda: fix published date bounding box, when there is author name bounding box in it
    if website_name == "pravda":
        if (author_bounding_box[0] >= date_bounding_box[0] and author_bounding_box[1] >= date_bounding_box[1]
                and author_bounding_box[2] - 1 <= date_bounding_box[2]
                and author_bounding_box[3] - 1 <= date_bounding_box[3]):
            date_bounding_box[1] = author_bounding_box[3]

    # Sme: remove parent references (
    if website_name == "sme":
        bounding_boxes_value['parent_reference'] = None

    bounding_boxes_value['date_published'] = date_bounding_box

    return bounding_boxes_value


def main():
    extended_output_data = "/tmp/knn/crawled-data-v2/extended_output_data" if len(sys.argv) < 2 else sys.argv[1]

    for website_path in glob.glob(f"{extended_output_data}/*"):
        website_name_search = re.search(r"/([a-z0-9]+)$", website_path)
        website_name = website_name_search.group(1)

        for article_path in glob.glob(f"{website_path}/*"):
            for page_path in glob.glob(f"{article_path}/bounding-boxes/*.pickle"):
                with open(page_path, 'rb') as f:
                    page_bounding_boxes: Dict[str, Dict[str, List[float]]] = pickle.load(f)

                for comment_id, comment_bounding_boxes in page_bounding_boxes.items():
                    # Fix known overlapping (of bounding boxes) issues
                    comment_bounding_boxes: Dict[str, List[float]] = apply_higher_level_fixes(
                        website_name,
                        comment_bounding_boxes
                    )

                    for section_name, bounding_box in comment_bounding_boxes.items():
                        if bounding_box is None:
                            continue

                        bounding_box = apply_offset(OFFSETS[website_name], bounding_box)

                        comment_bounding_boxes[section_name] = bounding_box

                    page_bounding_boxes[comment_id] = comment_bounding_boxes

                with open(page_path, 'wb') as f:
                    pickle.dump(page_bounding_boxes, f)


if __name__ == '__main__':
    main()
