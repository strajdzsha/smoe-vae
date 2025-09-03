from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import os
import requests
import numpy as np
import torch
from PIL import Image, ImageDraw
import math

# Custom Dataset for applying rotations
class RotatedMNISTDataset(Dataset):
    def __init__(self, base_dataset, rotation_degrees):
        self.base_dataset = base_dataset
        # Ensure rotation_degrees is a list, even if empty, for consistent handling
        self.rotation_degrees = rotation_degrees if isinstance(rotation_degrees, list) else []
        
        # Each original image will have 1 (original) + len(self.rotation_degrees) versions
        self.num_versions_per_image = 1 + len(self.rotation_degrees)
        if self.num_versions_per_image == 1 and len(self.rotation_degrees) == 0: # Only originals if no rotations
            pass # num_versions_per_image is correctly 1
        elif len(self.rotation_degrees) == 0: # Should not happen if logic is sound, but as a fallback
             self.num_versions_per_image = 1


    def __len__(self):
        # If no rotations are specified, length is just the base dataset length
        if not self.rotation_degrees:
            return len(self.base_dataset)
        return len(self.base_dataset) * self.num_versions_per_image

    def __getitem__(self, idx):
        if not self.rotation_degrees or self.num_versions_per_image == 1: # No rotations, behave like base
            return self.base_dataset[idx]

        original_image_idx = idx // self.num_versions_per_image
        version_idx = idx % self.num_versions_per_image # 0 for original, 1+ for rotations

        # Get the original image and label
        # The base_dataset should already have ToTensor applied.
        img, label = self.base_dataset[original_image_idx]

        if version_idx == 0:
            # This is the original image
            return img, label
        else:
            # This is a rotated version
            # Subtract 1 because version_idx=0 is original, so rotations start at version_idx=1
            rotation_angle = self.rotation_degrees[version_idx - 1]
            # TF.rotate expects a PIL Image or a Tensor. MNIST dataset with ToTensor returns a Tensor.
            rotated_img = TF.rotate(img, angle=float(rotation_angle))
            return rotated_img, label

# Combined Dataset that can include MNIST, FashionMNIST, CIFAR10, QuickDraw, etc.
class CombinedDataset(Dataset):
    def __init__(self, datasets=None, convert_cifar_to_grayscale=True, resize_to=28):
        """
        Creates a combined dataset that merges multiple datasets.
        
        Args:
            datasets: Dict of datasets with format {'name': dataset_object}
                     Supported names: 'mnist', 'fashion_mnist', 'cifar10', 'quickdraw'
            convert_cifar_to_grayscale: If True, converts CIFAR10 images to grayscale
            resize_to: Size to resize images to (default 28px for MNIST compatibility)
        """
        self.datasets = datasets or {}
        self.convert_cifar_to_grayscale = convert_cifar_to_grayscale
        self.resize_to = resize_to
        
        # Calculate dataset lengths for indexing
        self.lengths = {}
        self.offsets = {}
        self.starting_indices = {}
        
        # Dataset label offsets - each dataset gets 10 labels
        dataset_offsets = {
            'mnist': 0,
            'fashion_mnist': 10, 
            #'cifar10': 20,
            'quickdraw': 20
        }
        
        # Calculate starting indices and lengths
        current_idx = 0
        for name, dataset in self.datasets.items():
            self.lengths[name] = len(dataset)
            self.offsets[name] = dataset_offsets.get(name, 0)
            self.starting_indices[name] = current_idx
            current_idx += self.lengths[name]
        
        # Total labels in the combined dataset
        self.num_classes = 0
        for name, dataset in self.datasets.items():
            if name == 'quickdraw':
                self.num_classes += len(dataset.categories)
            else:
                self.num_classes += 10  # Standard datasets have 10 classes

    def __len__(self):
        return sum(self.lengths.values())

    def __getitem__(self, idx):
        # Determine which dataset to use based on the index
        for name, dataset in self.datasets.items():
            if idx < self.starting_indices[name] + self.lengths[name]:
                dataset_idx = idx - self.starting_indices[name]
                img, label = dataset[dataset_idx]
                
                # Add appropriate label offset
                label = label + self.offsets[name]
                
                # Special processing for CIFAR10 (RGB to grayscale conversion)
                if name == 'cifar10' and self.convert_cifar_to_grayscale:
                    # Convert RGB to grayscale: 0.2989 * R + 0.5870 * G + 0.1140 * B
                    img = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
                    # Reshape to match MNIST format: [1, H, W]
                    img = img.unsqueeze(0)
                
                    # CIFAR10 is 32x32, resize to match MNIST if needed
                    if self.resize_to != 32:
                        img = TF.resize(img, (self.resize_to, self.resize_to))
                
                return img, label
        
        raise IndexError(f"Index {idx} out of range for combined dataset with length {len(self)}")

# QuickDraw Image Dataset - renders QuickDraw sketches as 28x28 images
class QuickDrawImageDataset(Dataset):
    # Default categories to load
    DEFAULT_CATEGORIES = ['face', 'cat', 'snowflake', 'pencil', 'eye']
    EXPANDED_CATEGORIES = ['airplane', 'apple', 'banana', 'basketball', 'bread', 'cactus', 'car', 'cat', 'chair', 'clock', 'crown', 'envelope', 'fish', 'guitar', 'hammer', 'hat', 'ice cream', 'key', 'moon', 'tree']
    BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/"

    def __init__(self, root="data/quickdraw", categories=None, train=True, 
                 img_size=28, samples_per_category=10000, download=True, transform=None):
        """
        Creates a dataset of QuickDraw sketches rendered as images.
        
        Args:
            root: Directory where QuickDraw .npz files are stored
            categories: List of QuickDraw categories to load (default: DEFAULT_CATEGORIES)
            train: Data split to use ('train' or 'test')
            img_size: Size of output images (default: 28x28 to match MNIST)
            samples_per_category: Maximum samples to load per category
            download: If True, download missing datasets
            transform: Optional transform to apply to images
        """
        self.root = os.path.join(root, "quickdraw")
        self.train = train
        self.split = 'train' if train else 'test'
        self.img_size = img_size
        self.samples_per_category = samples_per_category
        self.transform = transform
        
        # Use default categories if none specified
        self.categories = categories if categories is not None else self.DEFAULT_CATEGORIES
        self.category_to_idx = {name: i for i, name in enumerate(self.categories)}
        
        self.images = []  # Will store (image_tensor, category_idx) tuples
        
        os.makedirs(self.root, exist_ok=True)
        
        # Load data for each category
        for cat_idx, category_name in enumerate(self.categories):
            file_path = self._download_if_needed(category_name, download)
            if file_path and os.path.exists(file_path):
                self._load_category(file_path, cat_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor, label = self.images[idx]
        return img_tensor, label

    def _download_if_needed(self, category_name, download):
        """Download the QuickDraw category file if needed."""
        category_formatted = category_name.replace(' ', '_').lower()
        file_path = os.path.join(self.root, f"{category_formatted}.npz")
        
        if os.path.exists(file_path):
            return file_path
        
        if not download:
            print(f"Category {category_name} not found and download=False.")
            return None
        
        url = f"{self.BASE_URL}{category_formatted}.npz"
        try:
            print(f"Downloading {category_name} from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved {category_name} to {file_path}")
            return file_path
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {category_name}: {e}")
            return None

    def _load_category(self, file_path, cat_idx):
        """Load and process a category's sketches."""
        try:
            data = np.load(file_path, allow_pickle=True, encoding='bytes')
            if self.split not in data:
                print(f"Warning: Split '{self.split}' not found in {file_path}. Skipping category.")
                return
            
            # Get strokes data for this category
            strokes = data[self.split]
            strokes = strokes[:self.samples_per_category]
            
            for stroke in strokes:
                if len(stroke) > 0:  # Ensure non-empty stroke
                    # Convert strokes to image
                    img_tensor = self._stroke_to_image(stroke)
                    self.images.append((img_tensor, cat_idx))
                    
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    def _stroke_to_image(self, stroke_data):
        """Convert a stroke sequence to a rasterized image with enhanced smoothness."""
        # Create a blank image with white background at higher resolution for anti-aliasing
        scale_factor = 2  # Higher resolution for better anti-aliasing
        large_size = self.img_size * scale_factor
        img = Image.new('L', (large_size, large_size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Scale factor to fit the sketch in the image
        # First, find min/max values from stroke data
        if len(stroke_data) <= 1:  # Need at least 2 points to draw
            return torch.zeros(1, self.img_size, self.img_size)
        
        # Extract absolute coordinates
        abs_x, abs_y = 0, 0
        points = []
        strokes = []
        current_stroke = []
        
        for i in range(len(stroke_data)):
            dx, dy, pen_state = stroke_data[i]
            abs_x += dx
            abs_y += dy
            points.append((abs_x, abs_y, pen_state))
            
            # Group points by stroke for better rendering
            current_stroke.append((abs_x, abs_y))
            if pen_state == 1:  # Pen up - end of stroke
                if len(current_stroke) > 1:  # Only add strokes with at least 2 points
                    strokes.append(current_stroke)
                current_stroke = []
        
        # Add the last stroke if it exists
        if len(current_stroke) > 1:
            strokes.append(current_stroke)
        
        # Find min/max coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Calculate scale to fit in image (with padding)
        width = max_x - min_x
        height = max_y - min_y
        
        # Avoid division by zero if sketch is a single point
        if width == 0 and height == 0:
            scale = 1.0
        elif width == 0:
            scale = (large_size * 0.8) / height
        elif height == 0:
            scale = (large_size * 0.8) / width
        else:
            scale = min((large_size * 0.8) / width, (large_size * 0.8) / height)
        
        # Center the sketch
        offset_x = large_size / 2 - (min_x + max_x) / 2 * scale
        offset_y = large_size / 2 - (min_y + max_y) / 2 * scale
        
        # Calculate line width based on image size (thicker for better visibility)
        line_width = max(3, large_size // 28)  # Minimum width of 3 pixels, scales with image size
        
        # Draw each stroke as a smooth curve
        for stroke in strokes:
            # Scale points to fit the image
            scaled_points = [(p[0] * scale + offset_x, p[1] * scale + offset_y) for p in stroke]
            
            # Draw smooth line segments
            if len(scaled_points) > 2:
                # Draw lines with anti-aliasing
                for i in range(len(scaled_points) - 1):
                    p1 = scaled_points[i]
                    p2 = scaled_points[i + 1]
                    
                    # Calculate distance between points
                    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    
                    # For longer segments, draw as smooth curves with more intermediate points
                    if dist > 10:
                        # Draw multiple sub-segments for better smoothness
                        steps = int(dist / 5)
                        for j in range(steps):
                            t1 = j / steps
                            t2 = (j + 1) / steps
                            sub_p1 = (p1[0] + t1 * (p2[0] - p1[0]), p1[1] + t1 * (p2[1] - p1[1]))
                            sub_p2 = (p1[0] + t2 * (p2[0] - p1[0]), p1[1] + t2 * (p2[1] - p1[1]))
                            draw.line([sub_p1, sub_p2], fill=0, width=line_width)
                    else:
                        draw.line([p1, p2], fill=0, width=line_width)
            elif len(scaled_points) == 2:
                # Simple line for just two points
                draw.line(scaled_points, fill=0, width=line_width)
        
        # Resize back to target size with anti-aliasing
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Convert to tensor format - single channel image
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
        
        return img_tensor
    