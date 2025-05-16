# Synthetic2Real Object Detection Challenge 2

This repository contains code and utilities for the [Synthetic2Real Object Detection Challenge 2](https://www.kaggle.com/competitions/synthetic-2-real-object-detection-challenge-2/data) on Kaggle. The goal is to train a model that predicts bounding box coordinates for objects in images, using only synthetic data.

## Project Structure

```
_models.py
main.ipynb
view.ipynb
checkpoint.pt
sample_submission.csv
training_progress.png
train/
val/
testImages/
annotated_real_data/
```

- **_models.py**: Core model, dataset, training, evaluation, and visualization code.
- **main.ipynb**: Notebook for training and evaluating the model.
- **view.ipynb**: Notebook for visualizing predictions on images.
- **checkpoint.pt**: Saved model weights.
- **sample_submission.csv**: Example submission file for Kaggle.
- **training_progress.png**: Training/validation loss plot.
- **train/**, **val/**: Each contains images (`.jpg` or `.png`) and corresponding label files (e.g., `.txt` or `.csv`) for each image.  
  Example:
  
  ```
  train/
    base/
        images/
            img_001.jpg
            img_002.jpg
            ...
        labels/
            img_001.txt
            img_002.txt
            ...
    cameraDistance/
        images/
            img_001.jpg
            img_002.jpg
            ...
        labels/
            img_001.txt
            img_002.txt
            ...
    ...
  ```
  The `val/` and `testImages/`  folder follow the same structure as `train/base/`.
  
- **(Optional) annotated_real_data/**: Additional real annotated data.

## Setup

1. **Install dependencies**  
   This project requires Python 3.12+ and the following packages:
   
   - torch
   - torchvision
   - matplotlib
   - numpy
   - tqdm
   - pillow
   
   Install with pip:
   ```sh
   pip install torch torchvision matplotlib numpy tqdm pillow
   ```
   
2. **Download and organize data**  
   Download the competition data from Kaggle and extract it into the appropriate folders (`train/`, `val/`, `testImages/`, etc.) as per the competition instructions.

## Training

To train the model, open and run the cells in [main.ipynb](main.ipynb). This will:

- Load the datasets
- Initialize the model
- Train with early stopping
- Save the best model to `checkpoint.pt`
- Plot training and validation loss

## Evaluation

After training, the notebook evaluates the model on the test set and prints the final loss.

## Visualization

To visualize predictions, use [view.ipynb](view.ipynb). This notebook:

- Loads a trained model
- Displays images with predicted and (optionally) ground truth bounding boxes

## Model

The model is a simple convolutional neural network defined in [`CoordinateCNN`](./_models.py). It predicts the normalized center coordinates and width/height of the bounding box for each image.

## License

This project is licensed under the GNU GPL v3. See [LICENSE](LICENSE) for details.

---

**Contact:**  
For questions or issues, please open an issue on this repository.
