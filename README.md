# AugmentaCV: Automated Image Augmentation Pipeline

A modular Python and OpenCV pipeline for spatial-domain enhancement and geometric transformations.

## Project Structure
* `transforms/spatial.py`: Intensity transformations (brightness, contrast, negative, log, gamma, thresholding, contrast stretching).
* `transforms/geometric.py`: Spatial coordinate transforms (translation, scaling, rotation, reflections, shearing, affine mapping).
* `main.py`: Pipeline execution and processing engine.
* `test_pipeline.py`: Automated unit tests for dimension and dtype validation.

## Requirements
* Python 3.8+
* OpenCV (`opencv-python`)
* NumPy

## Installation
```bash
pip install -r requirements.txt
