# Dependencies

You can use Conda for installing all the required dependencies.

CPU version (suitable for dataset (pre)processing):

`conda create -n knn-cpu -y python=3.11 pytorch-cpu detectron2 transformers datasets pillow pandas seqeval pyyaml matplotlib jupyter pytesseract`

GPU version (suitable for model training):

`conda create -n knn-gpu -y python=3.11 pytorch-gpu detectron2 transformers datasets pillow pandas seqeval pyyaml matplotlib jupyter pytesseract`
