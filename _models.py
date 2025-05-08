import os

import matplotlib.pyplot as plt
import torch
from PIL import Image as PILImage
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CoordinateDataset(Dataset):

    def __init__(
        self, img_dir: str, label_dir: str, transform: transforms.Compose = None
    ):
        """
        Args:
            img_dir (string): Directory with all the images
            label_dir (string): Directory with all the label text files
            transform (callable, optional): Optional transform to be applied on images
        """
        self.__img_dir: str = img_dir
        self.__label_dir: str = label_dir
        self.__transform: transforms.Compose = transform

        # Get sorted list of files to ensure proper pairing
        self.__img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".png")]
        )

    @staticmethod
    def load_coordinates(label_path: str) -> torch.Tensor:
        """
        Load coordinates from a label file.
        Args:
            label_path (string): Path to the label file
        Returns:
            torch.Tensor: Coordinates as a tensor
        """
        with open(label_path, "r") as f:
            line = f.readline().strip()
            parts = line.split(" ")
            coords = [float(p) for p in parts]
        return torch.tensor(coords, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.__img_files)

    def __getitem__(self, idx: int) -> tuple[PILImage.Image, torch.Tensor]:
        # Get image path
        img_path = os.path.join(self.__img_dir, self.__img_files[idx])

        # Get corresponding label path (replace image extension with .txt)
        label_name = os.path.splitext(self.__img_files[idx])[0] + ".txt"
        label_path = os.path.join(self.__label_dir, label_name)

        # Load image
        image: PILImage.Image = PILImage.open(img_path).convert("RGB")

        # Apply transformations if any
        if self.__transform:
            image = self.__transform(image)

        # Convert coordinates to tensor without normalization
        coords = self.load_coordinates(label_path)

        return image, coords


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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Training function
def train_model(
    device: torch.device,
    model: CoordinateCNN,
    dataloader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int = 10,
) -> None:
    model.train()

    for epoch in range(num_epochs):
        running_loss: float = 0.0

        for images, coords in dataloader:
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

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Visualization of predictions
def visualize_prediction(
    image_path: str,
    model: torch.nn.Module,
    transform: transforms.Compose = None,
    true_coords: torch.Tensor = None,
) -> None:
    # Set default transform if not provided
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

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
