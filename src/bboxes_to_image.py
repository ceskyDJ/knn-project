import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# TODO uprav kod aby to slo viac automaticky (rozbalenie adresarov a nasledne prelozenie viacerych screenov + prvotny check
# ci to neumiestnuje bb na rovnake miesta)

nameOfFile = "zive/5"
numberOfScreenshotAndPickle = "1"

with open(f'{nameOfFile}/bounding-boxes/{numberOfScreenshotAndPickle}.pickle', 'rb') as f:
    bounding_boxes_data = pickle.load(f)

image = Image.open(f'{nameOfFile}/screenshot/{numberOfScreenshotAndPickle}.png')

fig, ax = plt.subplots(figsize=(785/100, 2015/100))
ax.imshow(image)
ax.axis('off')

# print(bounding_boxes_data.items())

for bounding_boxes_key, bounding_boxes_value in bounding_boxes_data.items():
    x, y, w, h = bounding_boxes_value['wrapper']
    rect = patches.Rectangle((x, y), (w-x), (h-y), linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, 'wrapper', fontsize=5, color='red', verticalalignment='bottom')

    x, y, w, h = bounding_boxes_value['id']
    rect = patches.Rectangle((x, y), (w-x), (h-y), linewidth=1, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, 'id', fontsize=5, color='green', verticalalignment='bottom')

    x, y, w, h = bounding_boxes_value['date_published']
    rect = patches.Rectangle((x, y), (w-x), (h-y), linewidth=1, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, 'date_published', fontsize=5, color='blue', verticalalignment='bottom')

    x, y, w, h = bounding_boxes_value['text']
    rect = patches.Rectangle((x, y), (w-x), (h-y), linewidth=1, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, 'text', fontsize=5, color='yellow', verticalalignment='bottom')
    
    x, y, w, h = bounding_boxes_value['id']
    rect = patches.Rectangle((x, y), (w-x), (h-y), linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, 'author_id', fontsize=5, color='black', verticalalignment='bottom')

    x, y, w, h = bounding_boxes_value['author_name']
    rect = patches.Rectangle((x, y), (w-x), (h-y), linewidth=1, edgecolor='brown', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, 'author_name', fontsize=5, color='brown', verticalalignment='bottom')
    
    # print(bounding_boxes_value)

output_directory = f'{nameOfFile}/screenshot_with_bb/'
os.makedirs(output_directory, exist_ok=True)

plt.savefig(f'{nameOfFile}/screenshot_with_bb/{numberOfScreenshotAndPickle}_bb.png', dpi=300, bbox_inches='tight', pad_inches=0)