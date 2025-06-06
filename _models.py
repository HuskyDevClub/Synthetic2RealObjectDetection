import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from tqdm import tqdm


def get_all_images_pathlib(folder_path: str) -> list[str]:
    # Common image file extensions
    image_extensions: tuple = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    )

    # Convert string path to Path object
    folder: Path = Path(folder_path)

    # Use glob pattern to recursively find all files, then filter for images
    image_files: list[str] = sorted(
        str(f)
        for f in folder.glob("**/*")
        if f.is_file() and f.suffix.lower() in image_extensions
    )

    return image_files


class CoordinateDataset(Dataset):

    # SIZE: tuple[int, int] = (512, 512)  # Size of the images
    SIZE: tuple[int, int] = (256, 256)  # Size of the images

    # Define the transformation sequence
    __TRANSFORM_SEQUENCE: list = [
        transforms.Resize(SIZE),
        transforms.ToTensor(),
    ]

    # If RGB is disabled, convert to Grayscale
    ENABLE_RGB: bool = True  # Use RGB images if True, Grayscale if False
    if not ENABLE_RGB:
        __TRANSFORM_SEQUENCE.insert(0, transforms.Grayscale(num_output_channels=1))

    # Train with transforms
    ENABLE_TRANSFORM_TRAIN: bool = True

    # Define transformations (without normalization for images)
    TRANSFORM: transforms.Compose = transforms.Compose(__TRANSFORM_SEQUENCE)

    # Update hyperparameters based on the model path
    @classmethod
    def update_hyperparameters(cls, path: str) -> None:
        if not path.endswith(".pt"):
            raise ValueError("Path must point to a .pt file")

        # Extract filename to determine RGB
        filename: str = os.path.basename(path)
        cls.ENABLE_RGB = "rbg" in filename
        if cls.ENABLE_RGB:
            if len(cls.__TRANSFORM_SEQUENCE) == 3:
                cls.__TRANSFORM_SEQUENCE.pop(0)  # Remove Grayscale if it exists
        elif len(cls.__TRANSFORM_SEQUENCE) == 2:
            cls.__TRANSFORM_SEQUENCE.insert(
                0, transforms.Grayscale(num_output_channels=1)
            )

        # Extract width from filename (assumes format like "dataset_rbg_256.pt")
        w: int = int(filename.strip(".pt").split("_")[-1])
        cls.SIZE = (w, w)
        cls.__TRANSFORM_SEQUENCE[-2] = transforms.Resize(cls.SIZE)

        # Update the transformation
        cls.TRANSFORM = transforms.Compose(cls.__TRANSFORM_SEQUENCE)

        # Enable or disable transforms based on filename
        cls.ENABLE_TRANSFORM_TRAIN = "light" not in filename

    def __init__(self, dataset_dir: str):
        """
        Args:
            img_dir (string): Directory with all the images
            label_dir (string): Directory with all the label text files
            transform (callable, optional): Optional transform to be applied on images
        """
        self.__dataset_dir: str = dataset_dir

        # Get sorted list of files to ensure proper pairing
        img_files: list[str] = list(
            filter(
                lambda f: self.ENABLE_TRANSFORM_TRAIN or "_transform_" not in f,
                get_all_images_pathlib(self.__dataset_dir),
            )
        )

        # Load coordinates from label files
        self.__coords: list[torch.Tensor] = []
        for i in range(len(img_files) - 1, -1, -1):
            # Get corresponding label path (replace image extension with .txt)
            label_name = os.path.splitext(os.path.basename(img_files[i]))[0] + ".txt"
            label_path = os.path.join(
                os.path.dirname(os.path.dirname(img_files[i])),
                "labels",
                label_name,
            )

            # Load coordinates
            coords: torch.Tensor | None = self.load_coordinates(label_path)
            # Remove image if label is invalid
            if coords is None:
                os.remove(img_files.pop(i))
            else:
                self.__coords.append(coords)
        self.__coords.reverse()

        assert len(img_files) == len(self.__coords)

        self.__images: list[PILImage.Image] = [
            PILImage.open(f)
            .convert("RGB" if self.ENABLE_RGB else "L")
            .resize(self.SIZE)
            for f in img_files
        ]

    @staticmethod
    def load_coordinates(label_path: str) -> torch.Tensor | None:
        """
        Load coordinates from a label file.
        Args:
            label_path (string): Path to the label file
        Returns:
            torch.Tensor: Coordinates as a tensor
        """

        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Label file {label_path} does not exist.")
            return None

        # Read the label files
        with open(label_path, "r") as f:
            line = f.readline().strip()
            parts = line.split(" ")
            try:
                coords = [float(p) for p in parts]
                assert len(coords) == 5, "Expected 5 values in the label file"
                assert sum(coords) > 0
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
                return None
        return torch.tensor(coords[1:], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, idx: int) -> tuple[PILImage.Image, torch.Tensor]:
        return self.TRANSFORM(self.__images[idx]), self.__coords[idx]


# A simple CNN model for coordinate prediction
class CoordinateCNN(nn.Module):
    def __init__(self, num_coords: int = 4) -> None:
        super(CoordinateCNN, self).__init__()

        self.features = nn.Sequential(
            # First block - 32 filters
            nn.Conv2d(
                3 if CoordinateDataset.ENABLE_RGB else 1,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block - 64 filters
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third block - 128 filters
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth block - 256 filters
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                CoordinateDataset.SIZE[0] * CoordinateDataset.SIZE[1],
                CoordinateDataset.SIZE[0],
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(CoordinateDataset.SIZE[0], 128),
            nn.ReLU(inplace=True),
        )

        # Separate heads for coordinates and confidence
        self.coord_head = nn.Linear(128, num_coords)
        self.conf_head = nn.Sequential(
            nn.Linear(128, 1), nn.Sigmoid()  # Confidence between 0 and 1
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        coords = self.coord_head(x)
        confidence = self.conf_head(x)

        return coords, confidence


# Modified loss function that includes confidence
class DetectionLoss(nn.Module):
    def __init__(self, coord_weight=1.0, conf_weight=1.0):
        super(DetectionLoss, self).__init__()
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred_coords, pred_conf, true_coords, target_conf=None):
        # Coordinate loss
        coord_loss = self.mse_loss(pred_coords, true_coords)

        # If target confidence not provided, calculate based on IoU
        if target_conf is None:
            with torch.no_grad():
                # Calculate IoU between predicted and true boxes
                target_conf = self.calculate_iou_confidence(pred_coords, true_coords)

        # Confidence loss
        conf_loss = self.bce_loss(pred_conf, target_conf)

        # Combined loss
        total_loss = self.coord_weight * coord_loss + self.conf_weight * conf_loss

        return total_loss, coord_loss, conf_loss

    def calculate_iou_confidence(self, pred_coords, true_coords):
        """Calculate IoU-based confidence targets"""
        batch_size = pred_coords.shape[0]
        ious = []

        for i in range(batch_size):
            # Convert center format to corner format
            pred = pred_coords[i]
            true = true_coords[i]

            # Calculate IoU (simplified version)
            # You can use the more detailed IoU calculation from your existing code
            iou = self.calculate_single_iou(pred, true)
            ious.append(iou)

        return torch.tensor(ious, device=pred_coords.device).unsqueeze(1)

    def calculate_single_iou(self, pred, true):
        """Simplified IoU calculation for single box pair"""
        # Convert from center format to corner format
        pred_x1 = pred[0] - pred[2] / 2
        pred_y1 = pred[1] - pred[3] / 2
        pred_x2 = pred[0] + pred[2] / 2
        pred_y2 = pred[1] + pred[3] / 2

        true_x1 = true[0] - true[2] / 2
        true_y1 = true[1] - true[3] / 2
        true_x2 = true[0] + true[2] / 2
        true_y2 = true[1] + true[3] / 2

        # Calculate intersection
        x1 = max(pred_x1, true_x1)
        y1 = max(pred_y1, true_y1)
        x2 = min(pred_x2, true_x2)
        y2 = min(pred_y2, true_y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
        union = pred_area + true_area - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-6)

        return iou.item()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="models/checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'models/checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def convert_to_detection_format(
    coords: torch.Tensor, image_size: tuple[int, int] = (512, 512)
) -> dict:
    """
    Convert normalized coordinates to detection format for torchmetrics.

    Args:
        coords: Tensor of shape [batch_size, 4] with normalized coords (x_center, y_center, width, height)
        image_size: (height, width) of images

    Returns:
        Dictionary with boxes, scores, and labels in the format expected by torchmetrics
    """
    batch_size = coords.shape[0]
    h, w = image_size

    # Convert from normalized center format to absolute corner format
    x_center = coords[:, 0] * w
    y_center = coords[:, 1] * h
    width = coords[:, 2] * w
    height = coords[:, 3] * h

    # Convert to corner format (x1, y1, x2, y2)
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Stack to create boxes tensor
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Create scores (all 1.0 for ground truth, can be model confidence for predictions)
    scores = torch.ones(batch_size)

    # Create labels (assuming single class detection, label=0)
    labels = torch.zeros(batch_size, dtype=torch.int)

    return {"boxes": boxes, "scores": scores, "labels": labels}


def convert_to_detection_format_with_confidence(
    coords: torch.Tensor, confidence: float, image_size: tuple[int, int] = (512, 512)
) -> dict:
    """Convert predictions to detection format with confidence scores"""
    h, w = image_size

    # Convert from normalized center format to absolute corner format
    x_center = coords[0, 0] * w
    y_center = coords[0, 1] * h
    width = coords[0, 2] * w
    height = coords[0, 3] * h

    # Convert to corner format
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    boxes = torch.tensor([[x1, y1, x2, y2]])
    scores = torch.tensor([confidence])
    labels = torch.tensor([0], dtype=torch.int)

    return {"boxes": boxes, "scores": scores, "labels": labels}


def evaluate_model(
    model, val_loader, device, criterion, description: str = "Validation"
) -> dict[str, float]:
    """
    Evaluate model and return detailed metrics.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        image_size: Image dimensions for mAP calculation
    Returns:
        Dictionary with evaluation metrics
    """

    # Validation phase
    model.eval()
    val_loss: float = 0.0
    val_coord_loss: float = 0.0
    val_conf_loss: float = 0.0

    val_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=description)

        for images, coords in val_pbar:
            images = images.to(device)
            coords = coords.to(device)

            # Forward pass
            pred_coords, pred_conf = model(images)
            loss, coord_loss, conf_loss = criterion(pred_coords, pred_conf, coords)

            # Update statistics
            val_loss += loss.item() * images.size(0)
            val_coord_loss += coord_loss.item() * images.size(0)
            val_conf_loss += conf_loss.item() * images.size(0)

            # Convert to detection format with confidence scores
            batch_size = images.size(0)
            preds = []
            targets = []

            for i in range(batch_size):
                # For predictions, use the model's confidence
                pred_dict = convert_to_detection_format_with_confidence(
                    pred_coords[i : i + 1].cpu(),
                    pred_conf[i : i + 1].cpu().squeeze(),
                    CoordinateDataset.SIZE,
                )
                # For targets, use confidence of 1.0
                target_dict = convert_to_detection_format(
                    coords[i : i + 1].cpu(), CoordinateDataset.SIZE
                )

                preds.append(pred_dict)
                targets.append(target_dict)

            val_metric.update(preds, targets)

            val_pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "coord": coord_loss.item(),
                    "conf": conf_loss.item(),
                }
            )

    # Calculate averages
    val_loss = val_loss / len(val_loader.dataset)
    val_coord_loss = val_coord_loss / len(val_loader.dataset)
    val_conf_loss = val_conf_loss / len(val_loader.dataset)

    # Calculate mAP metrics
    val_results = val_metric.compute()
    val_map: float = val_results["map"].item() * 100
    val_map50: float = val_results["map_50"].item() * 100
    val_map75: float = val_results["map_75"].item() * 100

    # return metrics
    return {
        "val_loss": val_loss,
        "val_coord_loss": val_coord_loss,
        "val_conf_loss": val_conf_loss,
        "val_map": val_map,
        "val_map50": val_map50,
        "val_map75": val_map75,
    }


def train_and_validate(
    device: torch.device,
    model: CoordinateCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int = 30,
    patience: int = 7,
) -> dict:
    """Train model with confidence scores"""

    best_map50 = 0.0
    patience_counter = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "train_coord_loss": [],
        "train_conf_loss": [],
        "val_loss": [],
        "val_coord_loss": [],
        "val_conf_loss": [],
        "val_map": [],
        "val_map50": [],
        "val_map75": [],
        "epochs": [],
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_coord_loss = 0.0
        train_conf_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, coords in train_pbar:
            images = images.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()

            # Forward pass - now returns coords and confidence
            pred_coords, pred_conf = model(images)

            # Calculate loss
            loss, coord_loss, conf_loss = criterion(pred_coords, pred_conf, coords)

            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * images.size(0)
            train_coord_loss += coord_loss.item() * images.size(0)
            train_conf_loss += conf_loss.item() * images.size(0)

            train_pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "coord": coord_loss.item(),
                    "conf": conf_loss.item(),
                }
            )

        # Calculate averages
        train_loss = train_loss / len(train_loader.dataset)
        train_coord_loss = train_coord_loss / len(train_loader.dataset)
        train_conf_loss = train_conf_loss / len(train_loader.dataset)

        # Validation phase
        val_result: dict[str, float] = evaluate_model(
            model,
            val_loader,
            device,
            criterion,
            description=f"Epoch {epoch+1}/{num_epochs} [Val]",
        )
        val_loss = val_result["val_loss"]
        val_coord_loss = val_result["val_coord_loss"]
        val_conf_loss = val_result["val_conf_loss"]
        val_map = val_result["val_map"]
        val_map50 = val_result["val_map50"]
        val_map75 = val_result["val_map75"]

        # Update history
        history["train_loss"].append(train_loss)
        history["train_coord_loss"].append(train_coord_loss)
        history["train_conf_loss"].append(train_conf_loss)
        history["val_loss"].append(val_loss)
        history["val_coord_loss"].append(val_coord_loss)
        history["val_conf_loss"].append(val_conf_loss)
        history["val_map"].append(val_map)
        history["val_map50"].append(val_map50)
        history["val_map75"].append(val_map75)
        history["epochs"].append(epoch + 1)

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.6f} (coord: {train_coord_loss:.6f}, conf: {train_conf_loss:.6f}), "
            f"Val Loss: {val_loss:.6f} (coord: {val_coord_loss:.6f}, conf: {val_conf_loss:.6f}), "
            f"Val mAP50: {val_map50:.2f}%"
        )

        # Early stopping
        if val_map50 > best_map50:
            best_map50 = val_map50
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "models/checkpoint.pt")
            print(f"New best mAP50: {best_map50:.2f}% - Model saved!")
        else:
            patience_counter += 1
            print(
                f"No improvement in mAP50. Current best mAP50: {best_map50:.2f} - Patience: {patience_counter}/{patience}"
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def plot_training_history(history: dict):
    """
    Plot the training and validation loss with mAP metrics.

    Args:
        history (dict): Training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot losses
    ax1.plot(
        history["epochs"], history["train_loss"], label="Training Loss", marker="o"
    )
    ax1.plot(
        history["epochs"], history["val_loss"], label="Validation Loss", marker="x"
    )
    ax1.set_title("Training and Validation MSE Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot mAP metrics
    ax2.plot(history["epochs"], history["val_map"], label="mAP", marker="o")
    ax2.plot(history["epochs"], history["val_map50"], label="mAP@50", marker="x")
    ax2.plot(history["epochs"], history["val_map75"], label="mAP@75", marker="s")
    ax2.set_title("Validation mAP Metrics")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("mAP (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.show()


def plot_coordinates(
    coords: torch.Tensor, w: int, h: int, label: str, color: str
) -> None:
    true_x, true_y, true_w, true_h = coords.cpu().numpy()
    true_x1 = int((true_x - true_w / 2) * w)
    true_y1 = int((true_y - true_h / 2) * h)
    true_x2 = int((true_x + true_w / 2) * w)
    true_y2 = int((true_y + true_h / 2) * h)
    plt.plot(
        [true_x1, true_x2, true_x2, true_x1, true_x1],
        [true_y1, true_y1, true_y2, true_y2, true_y1],
        color=color,
        linewidth=2,
        label=label,
    )


# Visualization of predictions
def visualize_prediction(
    image_path: str,
    model: torch.nn.Module,
    true_coords: torch.Tensor = None,
) -> None:
    # Load and transform the image
    image = PILImage.open(image_path).convert("RGB")
    original_width, original_height = image.size

    # Transform the image
    transformed_image = CoordinateDataset.TRANSFORM(image)

    # Get model device
    device = next(model.parameters()).device

    # Move input tensor to the same device as model
    transformed_image = transformed_image.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Get prediction from model
    with torch.no_grad():
        # Add batch dimension
        input_image = transformed_image.unsqueeze(0)

        # Get prediction
        pred_coords, pred_conf = model(input_image)  # Get first item from batch
        pred_coords = pred_coords.cpu()[0]  # Move to CPU for visualization

    # Convert image tensor to numpy for visualization (move back to CPU first)
    img = transformed_image.cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # Get dimensions of transformed image
    h, w = transformed_image.shape[1:]  # Height and width after transformation

    # Draw predicted bounding box (red)
    plot_coordinates(pred_coords, w, h, label="Prediction", color="red")
    plt.title("Confidence: {:.2f}".format(pred_conf.cpu().numpy()[0][0]))

    # Draw true bounding box (green) if provided
    if true_coords is not None:
        true_coords = true_coords.to(device)  # Move to same device if provided
        plot_coordinates(true_coords, w, h, label="True", color="green")

    plt.legend()
    plt.show()

    # Return original image dimensions and prediction for reference
    return {
        "original_size": (original_width, original_height),
        "prediction": pred_coords.cpu().numpy(),
    }


# Visualization of predictions
def prediction_as_YOLO(
    image_path: str,
    model: torch.nn.Module,
) -> np.ndarray:
    # Load and transform the image
    image = PILImage.open(image_path).convert("RGB")

    # Transform the image
    transformed_image = CoordinateDataset.TRANSFORM(image)

    # Get model device
    device = next(model.parameters()).device

    # Move input tensor to the same device as model
    transformed_image = transformed_image.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Get prediction from model
    with torch.no_grad():
        # Add batch dimension
        input_image = transformed_image.unsqueeze(0)

        # Get prediction
        result = model(input_image)
        pred_coords = result[0]
        confidence = result[1].cpu().numpy()[0, 0]

    x_center, y_center, width, height = pred_coords.cpu().numpy()[0]

    # Return original image dimensions and prediction for reference
    return confidence, x_center, y_center, width, height
