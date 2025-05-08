import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
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

    def __init__(self, dataset_dir: str, transform: transforms.Compose = None):
        """
        Args:
            img_dir (string): Directory with all the images
            label_dir (string): Directory with all the label text files
            transform (callable, optional): Optional transform to be applied on images
        """
        self.__dataset_dir: str = dataset_dir
        self.__transform: transforms.Compose = transform

        # Get sorted list of files to ensure proper pairing
        self.__img_files: list[str] = get_all_images_pathlib(self.__dataset_dir)

        # Load coordinates from label files
        self.__coords: list[torch.Tensor] = []
        for i in range(len(self.__img_files) - 1, -1, -1):
            # Get corresponding label path (replace image extension with .txt)
            label_name = (
                os.path.splitext(os.path.basename(self.__img_files[i]))[0] + ".txt"
            )
            label_path = os.path.join(
                os.path.dirname(os.path.dirname(self.__img_files[i])),
                "labels",
                label_name,
            )

            # Load coordinates
            coords: torch.Tensor | None = self.load_coordinates(label_path)
            # Remove image if label is invalid
            if coords is None:
                self.__img_files.pop(i)
            else:
                self.__coords.append(coords)
        self.__coords.reverse()

        # print(self.__img_files[-1])
        # print(self.__getitem__(-1))

        assert len(self.__img_files) == len(self.__coords)

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
        return torch.tensor(coords, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.__img_files)

    def __getitem__(self, idx: int) -> tuple[PILImage.Image, torch.Tensor]:
        # Load image
        image: PILImage.Image = PILImage.open(self.__img_files[idx]).convert("RGB")

        # Apply transformations if any
        if self.__transform:
            image = self.__transform(image)

        return image, self.__coords[idx]


# Define a simple CNN model for coordinate prediction
class CoordinateCNN(nn.Module):
    def __init__(
        self, num_coords: int = 5
    ) -> None:  # Default: class_id + 4 coordinates
        super(CoordinateCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 512),  # Adjust based on your input image size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_coords),
        )

        # Initialize weights using He initialization
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
        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
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


def create_data_loaders(
    full_dataset: CoordinateDataset,
    batch_size=32,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Calculate sizes
    test_count: int = int(test_size * len(full_dataset))
    train_val_count: int = len(full_dataset) - test_count
    val_count: int = int(val_size * train_val_count)
    train_count: int = train_val_count - val_count

    # Split the dataset
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_count, test_count],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(random_state),
    )

    # Create data loaders
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(
        f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


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
    """
    Train the model with validation and early stopping.

    Args:
        device (torch.device): Device to train on
        model (CoordinateCNN): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience

    Returns:
        dict: Training history
    """
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize history dictionary to store metrics
    history = {"train_loss": [], "val_loss": [], "epochs": []}

    # Start training
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, coords in train_pbar:
            images = images.to(device)
            coords = coords.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, coords)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({"loss": loss.item()})

        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

            for images, coords in val_pbar:
                images = images.to(device)
                coords = coords.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, coords)

                # Update statistics
                val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix({"loss": loss.item()})

        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch + 1)

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}"
        )

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

    # Calculate total training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Load the best model
    model.load_state_dict(torch.load("checkpoint.pt"))

    return history


def evaluate_model(
    device: torch.device, model: CoordinateCNN, test_loader: DataLoader, criterion
):
    """
    Evaluate the model on the test set.

    Args:
        device (torch.device): Device to evaluate on
        model (CoordinateCNN): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion: Loss function

    Returns:
        float: Test loss
    """
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")

        for images, coords in test_pbar:
            images = images.to(device)
            coords = coords.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, coords)

            # Update statistics
            test_loss += loss.item() * images.size(0)
            test_pbar.set_postfix({"loss": loss.item()})

    # Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}")

    return test_loss


def plot_training_history(history: dict):
    """
    Plot the training and validation loss.

    Args:
        history (dict): Training history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        history["epochs"], history["train_loss"], label="Training Loss", marker="o"
    )
    plt.plot(
        history["epochs"], history["val_loss"], label="Validation Loss", marker="x"
    )
    plt.title("Training and Validation MSE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.show()


# Visualization of predictions
def visualize_prediction(
    image_path: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    true_coords: torch.Tensor = None,
) -> None:
    # Load and transform the image
    image = PILImage.open(image_path).convert("RGB")
    original_width, original_height = image.size

    # Transform the image
    transformed_image = transform(image)

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
        pred_coords = model(input_image)[0]  # Get first item from batch

    # Convert image tensor to numpy for visualization (move back to CPU first)
    img = transformed_image.cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # Get dimensions of transformed image
    h, w = transformed_image.shape[1:]  # Height and width after transformation

    # Draw predicted bounding box (red)
    pred_class, pred_x, pred_y, pred_w, pred_h = pred_coords.cpu().numpy()
    pred_x1 = int((pred_x - pred_w / 2) * w)
    pred_y1 = int((pred_y - pred_h / 2) * h)
    pred_x2 = int((pred_x + pred_w / 2) * w)
    pred_y2 = int((pred_y + pred_h / 2) * h)
    plt.plot(
        [pred_x1, pred_x2, pred_x2, pred_x1, pred_x1],
        [pred_y1, pred_y1, pred_y2, pred_y2, pred_y1],
        "r-",
        linewidth=2,
        label="Prediction",
    )

    # Draw true bounding box (green) if provided
    if true_coords is not None:
        true_coords = true_coords.to(device)  # Move to same device if provided
        true_class, true_x, true_y, true_w, true_h = true_coords.cpu().numpy()
        true_x1 = int((true_x - true_w / 2) * w)
        true_y1 = int((true_y - true_h / 2) * h)
        true_x2 = int((true_x + true_w / 2) * w)
        true_y2 = int((true_y + true_h / 2) * h)
        plt.plot(
            [true_x1, true_x2, true_x2, true_x1, true_x1],
            [true_y1, true_y1, true_y2, true_y2, true_y1],
            "g-",
            linewidth=2,
            label="Ground Truth",
        )
        plt.title(
            f"True class: {int(true_class)}, Pred class: {int(round(pred_class))}"
        )
    else:
        plt.title(f"Predicted class: {int(round(pred_class))}")

    plt.legend()
    plt.show()

    # Return original image dimensions and prediction for reference
    return {
        "original_size": (original_width, original_height),
        "prediction": pred_coords.cpu().numpy(),
    }
