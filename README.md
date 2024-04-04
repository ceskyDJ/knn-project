# Comment classifier

## Installation

1. install `pytorch` according to their instructions: <https://pytorch.org/get-started/locally/>
2. install `detectronv2`: <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>
    - best to use the most recent version straight from their github
3. the rest of the requirements can be installed using `pip install -r requirements.txt`

## Datasets

The training dataset is currently created from youtube comments screenshots, which are hand annotated
in [label-studio](https://github.com/HumanSignal/label-studio) (currently using a local instance).

These annotations are exported in the `JSON-mini` format. Since `label-studio` adds a prefix to the uploaded
images, I also export the data in the `COCO` format, since it includes the renamed images. That is the only
reason for the `COCO` export. The actual annotations are taken from the `JSON-min` file.

The images which were used are in the `white_bg` directory.

## Model

After training, model checkpoints are saved in a nested directory inside `src/`.

## Tutorial/inspiration

The file `Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb` (from [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb))
were used as a template for finetuning the model.

