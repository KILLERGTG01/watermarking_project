import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import numpy as np
import gc
from tqdm import tqdm
import warnings
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy import fftpack

# Set multiprocessing method to spawn for better stability
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Ignore certain warnings to reduce noise
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="nn.functional.interpolate")

# Use GPU if available with memory management
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Clear cache at start
    torch.cuda.empty_cache()
    # Set device to highest compute capability
    max_memory_gpu = 0
    max_device_id = 0
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        if total_memory > max_memory_gpu:
            max_memory_gpu = total_memory
            max_device_id = i
    device = torch.device(f"cuda:{max_device_id}")
    print(f"Using CUDA device {max_device_id} with {max_memory_gpu/1024**3:.1f} GB memory")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Optimized HiDDeN model with reduced complexity
class OptimizedHiDDeNModel(nn.Module):
    def __init__(self, watermark_strength=0.8):
        super(OptimizedHiDDeNModel, self).__init__()
        self.watermark_strength = watermark_strength
        
        # Simplified watermark classifier (fewer layers, fewer filters)
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Reduced from 64 to 32 filters
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced from 128 to 64 filters
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))  # Removed one Conv2d layer
        )
        self.classifier_fc = nn.Linear(64, 1)  # Input size reduced from 256 to 64
        
        # Simplified encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),  # Reduced from 64 to 32 filters
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Reduced from 128->64 to 32->16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Reduced from 64 to 32 filters
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Reduced from 128->64 to 32->16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Move model to GPU if available
        self.to(device)

    # Classification with optimized memory usage
    def classify(self, image):
        features = self.classifier(image)
        features = features.view(features.size(0), -1)
        return torch.sigmoid(self.classifier_fc(features))

    # Encoding with adjustable watermark strength
    def encode(self, image, watermark):
        if watermark.shape[2:] != image.shape[2:]:
            watermark = F.interpolate(watermark, size=image.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([image, watermark], dim=1)
        encoded_image = image + self.watermark_strength * self.encoder(combined)
        encoded_image = torch.clamp(encoded_image, -1, 1)
        return encoded_image

    def decode(self, watermarked_image):
        extracted_watermark = self.decoder(watermarked_image)
        return extracted_watermark
    
    def remove_watermark(self, watermarked_image):
        extracted_watermark = self.decode(watermarked_image)
        clean_image = watermarked_image - extracted_watermark
        clean_image = torch.clamp(clean_image, -1, 1)
        return clean_image

# Safe image opening with support for various formats
def safe_open_image(path):
    """Safely open images with different formats including RGBA"""
    try:
        img = Image.open(path)
        if img.mode == 'RGBA':
            # Convert RGBA to RGB by compositing on white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha as mask
            return background
        elif img.mode != 'RGB':
            return img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        return None

# Add visualization function for dataset distribution
def visualize_dataset_distribution(train_dataset, val_dataset, output_dir="visualizations"):
    """
    Create bar graphs showing distribution of watermarked and unwatermarked images
    in both training and validation datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count watermarked and unwatermarked images in training set
    train_wm_count = len(train_dataset.watermarked_image_files)
    train_clean_count = len(train_dataset.clean_image_files)
    
    # Count watermarked and unwatermarked images in validation set
    val_wm_count = len(val_dataset.watermarked_image_files)
    val_clean_count = len(val_dataset.clean_image_files)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training set bar graph
    categories = ['Watermarked', 'Unwatermarked']
    counts = [train_wm_count, train_clean_count]
    ax1.bar(categories, counts, color=['#ff6b6b', '#4ecdc4'])
    ax1.set_title('Training Dataset Distribution')
    ax1.set_ylabel('Number of Images')
    for i, count in enumerate(counts):
        ax1.text(i, count + 0.1, str(count), ha='center')
    
    # Validation set bar graph
    counts = [val_wm_count, val_clean_count]
    ax2.bar(categories, counts, color=['#ff6b6b', '#4ecdc4'])
    ax2.set_title('Validation Dataset Distribution')
    ax2.set_ylabel('Number of Images')
    for i, count in enumerate(counts):
        ax2.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Dataset distribution visualization saved to {os.path.join(output_dir, 'dataset_distribution.png')}")

# Function to create confusion matrix
def create_confusion_matrix(true_labels, predicted_labels, output_dir="visualizations"):
    """
    Create and save a confusion matrix visualization.
    true_labels: List of ground truth labels (1 for watermarked, 0 for unwatermarked)
    predicted_labels: List of predicted labels (1 for watermarked, 0 for unwatermarked)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Set up plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Unwatermarked", "Watermarked"],
                yticklabels=["Unwatermarked", "Watermarked"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Watermark Detection")
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n----- Confusion Matrix Statistics -----")
    print(f"True Positives (correctly identified as watermarked): {tp}")
    print(f"True Negatives (correctly identified as unwatermarked): {tn}")
    print(f"False Positives (incorrectly identified as watermarked): {fp}")
    print(f"False Negatives (incorrectly identified as unwatermarked): {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"Confusion matrix visualization saved to {os.path.join(output_dir, 'confusion_matrix.png')}")
    
    return cm

# Function to create a heat correlation map
def create_correlation_heatmap(model, images, output_dir="visualizations"):
    """
    Create a correlation heatmap showing relationships between image features
    and watermark detection confidence.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features from a sample of images
    features = []
    predictions = []
    
    with torch.no_grad():
        for img in images:
            # Move image to device
            img_tensor = img.to(device)
            
            # Get model prediction
            pred = model.classify(img_tensor).cpu().item()
            predictions.append(pred)
            
            # Extract basic image statistics as features
            img_np = img.squeeze(0).cpu().numpy()
            img_np = (img_np * 0.5) + 0.5  # Denormalize
            
            # Calculate various image statistics
            means = img_np.mean(axis=(1, 2))  # Channel means
            stds = img_np.std(axis=(1, 2))    # Channel standard deviations
            
            # Calculate texture features (simplified)
            r_g_corr = np.corrcoef(img_np[0].flatten(), img_np[1].flatten())[0, 1]
            r_b_corr = np.corrcoef(img_np[0].flatten(), img_np[2].flatten())[0, 1]
            g_b_corr = np.corrcoef(img_np[1].flatten(), img_np[2].flatten())[0, 1]
            
            # Calculate some frequency domain features
            r_fft = np.abs(fftpack.fft2(img_np[0]))
            g_fft = np.abs(fftpack.fft2(img_np[1]))
            b_fft = np.abs(fftpack.fft2(img_np[2]))
            
            r_high_freq = r_fft[r_fft.shape[0]//2:, r_fft.shape[1]//2:].mean()
            g_high_freq = g_fft[g_fft.shape[0]//2:, g_fft.shape[1]//2:].mean()
            b_high_freq = b_fft[b_fft.shape[0]//2:, b_fft.shape[1]//2:].mean()
            
            # Combine features
            feature_vector = [
                means[0], means[1], means[2],  # RGB means
                stds[0], stds[1], stds[2],     # RGB standard deviations
                r_g_corr, r_b_corr, g_b_corr,  # Color correlations
                r_high_freq, g_high_freq, b_high_freq  # High frequency components
            ]
            features.append(feature_vector)
    
    # Create dataframe with features and prediction
    columns = [
        'R_Mean', 'G_Mean', 'B_Mean',
        'R_Std', 'G_Std', 'B_Std',
        'R_G_Corr', 'R_B_Corr', 'G_B_Corr',
        'R_HighFreq', 'G_HighFreq', 'B_HighFreq',
        'WM_Confidence'
    ]
    
    data = np.hstack([features, np.array(predictions).reshape(-1, 1)])
    df = pd.DataFrame(data, columns=columns)
    
    # Calculate correlation
    corr = df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Correlation heatmap saved to {os.path.join(output_dir, 'correlation_heatmap.png')}")
    
    return corr

# Optimized dataset class with better error handling
class WatermarkDataset(Dataset):
    def __init__(self, clean_dir, watermarked_dir=None, watermark_path=None, transform=None, is_train=True):
        """
        Args:
            clean_dir (string): Directory with non-watermarked/clean images.
            watermarked_dir (string): Directory with watermarked images.
            watermark_path (string): Path to watermark image to use for training.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): Whether this is training or validation set.
        """
        self.clean_dir = clean_dir
        self.watermarked_dir = watermarked_dir
        self.transform = transform
        self.is_train = is_train
        self.watermark_tensor = None
        
        # Get all clean image files
        self.clean_image_files = []
        if os.path.exists(clean_dir):
            self.clean_image_files = [f for f in os.listdir(clean_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            # Limit dataset size for faster processing if too large
            if len(self.clean_image_files) > 10000 and is_train:
                print(f"Large dataset detected: limiting to 10000 images for training")
                self.clean_image_files = self.clean_image_files[:10000]
        
        # Get all watermarked image files if directory is provided
        self.watermarked_image_files = []
        if watermarked_dir and os.path.exists(watermarked_dir):
            self.watermarked_image_files = [f for f in os.listdir(watermarked_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            # Limit dataset size for faster processing if too large
            if len(self.watermarked_image_files) > 10000 and is_train:
                print(f"Large dataset detected: limiting to 10000 images for training")
                self.watermarked_image_files = self.watermarked_image_files[:10000]
        
        # Load watermark once during initialization
        if watermark_path and os.path.exists(watermark_path):
            try:
                watermark_img = safe_open_image(watermark_path)
                if watermark_img and transform:
                    self.watermark_tensor = transform(watermark_img)
                print(f"Using watermark from: {watermark_path}")
            except Exception as e:
                print(f"Warning: Unable to load watermark: {e}")
                self.watermark_tensor = None
        
        print(f"Dataset initialized with {len(self.clean_image_files)} clean images and {len(self.watermarked_image_files)} watermarked images")
    
    def __len__(self):
        if self.is_train:
            # For training, use both clean and watermarked images
            return len(self.clean_image_files) + len(self.watermarked_image_files)
        else:
            # For validation, use all available images
            return len(self.clean_image_files) + len(self.watermarked_image_files)
    
    def __getitem__(self, idx):
        try:
            # Determine if we're loading a clean or watermarked image
            if idx >= len(self.clean_image_files):
                # This is a watermarked image
                if len(self.watermarked_image_files) == 0:
                    # No watermarked images available, wrap around to clean images
                    clean_idx = idx % len(self.clean_image_files)
                    img_name = os.path.join(self.clean_dir, self.clean_image_files[clean_idx])
                    image = safe_open_image(img_name)
                    
                    if image is None:
                        # Fallback for corrupted images
                        return self.__getitem__((idx + 1) % len(self))
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    # Use a blank watermark if none provided
                    if self.watermark_tensor is None:
                        watermark = torch.zeros_like(image)
                    else:
                        watermark = self.watermark_tensor
                    
                    return {
                        'image': image,
                        'watermark': watermark,
                        'has_watermark': torch.tensor([0.0], dtype=torch.float32)
                    }
                else:
                    # Load actual watermarked image
                    watermarked_idx = idx - len(self.clean_image_files)
                    watermarked_idx = watermarked_idx % len(self.watermarked_image_files)  # Handle overflow
                    img_name = os.path.join(self.watermarked_dir, self.watermarked_image_files[watermarked_idx])
                    image = safe_open_image(img_name)
                    
                    if image is None:
                        # Fallback for corrupted images
                        return self.__getitem__((idx + 1) % len(self))
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    # Use a blank watermark if none provided
                    if self.watermark_tensor is None:
                        watermark = torch.zeros_like(image)
                    else:
                        watermark = self.watermark_tensor
                    
                    return {
                        'image': image,
                        'watermark': watermark,
                        'has_watermark': torch.tensor([1.0], dtype=torch.float32)
                    }
            else:
                # This is a clean/non-watermarked image
                img_name = os.path.join(self.clean_dir, self.clean_image_files[idx])
                image = safe_open_image(img_name)
                
                if image is None:
                    # Fallback for corrupted images
                    return self.__getitem__((idx + 1) % len(self))
                
                if self.transform:
                    image = self.transform(image)
                
                # Use a blank watermark if none provided
                if self.watermark_tensor is None:
                    watermark = torch.zeros_like(image)
                else:
                    watermark = self.watermark_tensor
                
                return {
                    'image': image,
                    'watermark': watermark,
                    'has_watermark': torch.tensor([0.0], dtype=torch.float32)
                }
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            # Return a different sample in case of error
            return self.__getitem__((idx + 1) % len(self))

# Modified training function to generate visualizations
def optimized_train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, visualize=True, output_dir="visualizations"):
    """Optimized training function for the HiDDeN model with visualization support"""
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        
    # Use mixed precision training if CUDA is available
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Use a learning rate scheduler to speed up convergence
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Loss functions
    classifier_criterion = nn.BCELoss()
    reconstruction_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5  # Stop training if no improvement after this many epochs
    
    # Lists to track metrics for visualization
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Clear memory before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model.train()
        train_total_loss = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in progress_bar:
            # Move data to the right device
            images = batch['image'].to(device)
            watermarks = batch['watermark'].to(device)
            has_watermark = batch['has_watermark'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Use mixed precision training if available
            with autocast() if use_amp else nullcontext():
                # Forward pass - classification
                watermark_pred = model.classify(images)
                classifier_loss = classifier_criterion(watermark_pred, has_watermark)
                
                # Process based on ground truth
                watermarked_indices = has_watermark.squeeze() > 0.5
                clean_indices = ~watermarked_indices
                
                total_loss = classifier_loss
                
                # Process all images in a single forward pass where possible
                if watermarked_indices.sum() > 0:
                    watermarked_images = images[watermarked_indices]
                    clean_recovered = model.remove_watermark(watermarked_images)
                    decoded_watermarks = model.decode(watermarked_images)
                    
                    # Simplified loss calculation
                    removal_loss = 0.1 * torch.mean(torch.abs(
                        clean_recovered[:, :, 1:, :] - clean_recovered[:, :, :-1, :]
                    ))
                    decoder_loss = 0.1 * torch.mean((decoded_watermarks.mean(dim=[2, 3]) - 0.5) ** 2)
                    
                    total_loss = total_loss + removal_loss + decoder_loss
                
                if clean_indices.sum() > 0:
                    clean_images = images[clean_indices]
                    clean_watermarks = watermarks[clean_indices]
                    
                    encoded_images = model.encode(clean_images, clean_watermarks)
                    encoder_loss = reconstruction_criterion(encoded_images, clean_images)
                    
                    # Simplify this part - combine classification and reconstruction
                    decoded_watermarks = model.decode(encoded_images)
                    decoder_recovery_loss = reconstruction_criterion(decoded_watermarks, clean_watermarks)
                    
                    total_loss = total_loss + 0.5 * encoder_loss + 0.5 * decoder_recovery_loss
            
            # Backward pass and optimize with scaling if using AMP
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            train_total_loss += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}"})
        
        # Validation with fewer metrics for speed
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_progress_bar:
                images = batch['image'].to(device)
                has_watermark = batch['has_watermark'].to(device)
                
                watermark_pred = model.classify(images)
                batch_val_loss = classifier_criterion(watermark_pred, has_watermark)
                val_loss += batch_val_loss.item()
                
                # Calculate accuracy
                predicted = (watermark_pred > 0.5).float()
                batch_accuracy = (predicted == has_watermark).float().mean().item()
                val_accuracy += batch_accuracy
                
                # Update progress bar
                val_progress_bar.set_postfix({"val_loss": f"{batch_val_loss.item():.4f}", "acc": f"{batch_accuracy:.4f}"})
        
        # Average loss and accuracy
        avg_train_loss = train_total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        
        # Track metrics for visualization
        epochs.append(epoch + 1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Print simple metrics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {avg_val_accuracy:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "watermark_model_best.pth")
            print(f"New best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Load the best model before returning
                model.load_state_dict(torch.load("watermark_model_best.pth", map_location=device))
                break
    
    # Create training metrics visualization
    if visualize and len(epochs) > 1:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300)
        plt.close()
        
        print(f"Training metrics visualization saved to {os.path.join(output_dir, 'training_metrics.png')}")
    
    return model

# Batch processing for efficient GPU utilization
def process_image_batch(batch_files, input_dir, watermark_tensor, output_unwatermarked, output_watermarked, model, quality='medium'):
    """Process a batch of images at once on GPU"""
    batch_tensors = []
    original_sizes = []
    filenames = []
    
    # Collection for visualization
    true_labels = []
    predicted_labels = []
    
    # Prepare batch - determine processing resolution based on quality
    if quality == 'low':
        process_size = (64, 64)
        resize_method = Image.BILINEAR
    elif quality == 'medium':
        process_size = (128, 128)
        resize_method = Image.BILINEAR
    else:  # high
        process_size = (256, 256)
        resize_method = Image.LANCZOS
    
    # Create transform based on quality setting
    transform = transforms.Compose([
        transforms.Resize(process_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Prepare all valid images in batch
    for filename in batch_files:
        try:
            img_path = os.path.join(input_dir, filename)
            image = safe_open_image(img_path)
            
            if image is None:
                continue
                
            original_sizes.append(image.size)
            image_tensor = transform(image).unsqueeze(0)
            batch_tensors.append(image_tensor)
            filenames.append(filename)
            
            # Estimate true label from filename (heuristic)
            true_has_watermark = 'watermark' in filename.lower()
            true_labels.append(1 if true_has_watermark else 0)
        except Exception as e:
            print(f"Error preparing {filename}: {e}")
            continue
    
    if not batch_tensors:
        return [], true_labels, predicted_labels  # No valid images in this batch
    
    # Stack tensors into a batch
    batch = torch.cat(batch_tensors, dim=0).to(device)
    
    # Process batch
    results = []
    with torch.no_grad():
        # Classify all at once
        probs = model.classify(batch).squeeze()
        
        # Process each image based on classification
        for i, (prob, filename, original_size) in enumerate(zip(probs, filenames, original_sizes)):
            try:
                image_tensor = batch[i:i+1]
                confidence = abs(float(prob) - 0.5) * 2
                
                # Record prediction for confusion matrix
                has_watermark = prob > 0.5
                predicted_labels.append(1 if has_watermark else 0)
                
                # Skip low confidence predictions
                if confidence < 0.2:
                    results.append(f"Skipped {filename} (low confidence: {confidence:.2f})")
                    continue
                    
                if has_watermark:
                    # Remove watermark
                    cleaned_image = model.remove_watermark(image_tensor)
                    cleaned_image = cleaned_image * 0.5 + 0.5  # Denormalize
                    cleaned_image_pil = transforms.ToPILImage()(cleaned_image.squeeze(0).cpu())
                    cleaned_image_pil = cleaned_image_pil.resize(original_size, resize_method)
                    output_path = os.path.join(output_unwatermarked, filename)
                    cleaned_image_pil.save(output_path)
                    results.append(f"Removed watermark from: {filename} (confidence: {confidence:.2f})")
                else:
                    # Add watermark
                    watermarked_image = model.encode(image_tensor, watermark_tensor)
                    watermarked_image = watermarked_image * 0.5 + 0.5  # Denormalize
                    watermarked_image_pil = transforms.ToPILImage()(watermarked_image.squeeze(0).cpu())
                    watermarked_image_pil = watermarked_image_pil.resize(original_size, resize_method)
                    output_path = os.path.join(output_watermarked, filename)
                    watermarked_image_pil.save(output_path)
                    results.append(f"Added watermark to: {filename} (confidence: {confidence:.2f})")
            except Exception as e:
                results.append(f"Error processing {filename}: {e}")
    
    # Clean up GPU memory
    del batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results, true_labels, predicted_labels

# Optimized single image processing with quality settings
def optimized_process_single_image(filename, input_dir, watermark_img, output_unwatermarked, output_watermarked, model, quality='medium'):
    """Process a single image with quality settings"""
    
    # Determine processing resolution based on quality
    if quality == 'low':
        process_size = (64, 64)
        resize_method = Image.BILINEAR
    elif quality == 'medium':
        process_size = (128, 128)
        resize_method = Image.BILINEAR
    else:  # high
        process_size = (256, 256)
        resize_method = Image.LANCZOS
    
    img_path = os.path.join(input_dir, filename)

    try:
        # Load image safely
        image = safe_open_image(img_path)
        if image is None:
            return f"Error: Could not open {filename}"
            
        original_width, original_height = image.size
        
        # Use quality-based image size for processing
        transform = transforms.Compose([
            transforms.Resize(process_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Pre-scale watermark once
        watermark = watermark_img.resize(process_size, Image.BILINEAR)
        watermark_tensor = transform(watermark).unsqueeze(0).to(device)
        
        # Process image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Classify first - skip processing if confidence is low
            watermark_prob = model.classify(image_tensor)
            has_watermark = watermark_prob > 0.5
            confidence = abs(watermark_prob.item() - 0.5) * 2
            
            # Only process if confidence is above threshold (avoids processing ambiguous images)
            if confidence > 0.2:
                if has_watermark.item():
                    # Remove watermark
                    cleaned_image = model.remove_watermark(image_tensor)
                    cleaned_image = cleaned_image * 0.5 + 0.5  # Denormalize
                    cleaned_image_pil = transforms.ToPILImage()(cleaned_image.squeeze(0).cpu())
                    cleaned_image_pil = cleaned_image_pil.resize((original_width, original_height), resize_method)
                    output_path = os.path.join(output_unwatermarked, filename)
                    cleaned_image_pil.save(output_path)
                    return f"Removed watermark from: {filename} (confidence: {confidence:.2f})"
                else:
                    # Add watermark
                    watermarked_image = model.encode(image_tensor, watermark_tensor)
                    watermarked_image = watermarked_image * 0.5 + 0.5  # Denormalize
                    watermarked_image_pil = transforms.ToPILImage()(watermarked_image.squeeze(0).cpu())
                    watermarked_image_pil = watermarked_image_pil.resize((original_width, original_height), resize_method)
                    output_path = os.path.join(output_watermarked, filename)
                    watermarked_image_pil.save(output_path)
                    return f"Added watermark to: {filename} (confidence: {confidence:.2f})"
            else:
                return f"Skipped {filename} (low confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

# Modified process_in_chunks to track data for confusion matrix
def process_in_chunks(image_files, chunk_size=100, **kwargs):
    """Process large sets of files in manageable chunks"""
    total_processed = 0
    all_results = []
    all_true_labels = []
    all_predicted_labels = []
    
    for i in range(0, len(image_files), chunk_size):
        chunk = image_files[i:i+min(chunk_size, len(image_files)-i)]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(image_files)-1)//chunk_size + 1} ({len(chunk)} files)")
        
        # Process this chunk
        processed, chunk_true_labels, chunk_pred_labels = optimized_process_images(chunk, **kwargs)
        
        processed_count = sum(1 for r in processed if not r.startswith("Skipped") and not r.startswith("Error"))
        total_processed += processed_count
        all_results.extend(processed)
        all_true_labels.extend(chunk_true_labels)
        all_predicted_labels.extend(chunk_pred_labels)
        
        # Clear memory between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        
    return total_processed, all_results, all_true_labels, all_predicted_labels

# Modified main image processing function to support visualization
def optimized_process_images(image_files, input_dir, watermark_path, output_unwatermarked, output_watermarked, model, quality='medium', force_cpu=False, create_visualizations=True):
    """Process images with optimal resource usage and collect data for visualizations"""
    os.makedirs(output_unwatermarked, exist_ok=True)
    os.makedirs(output_watermarked, exist_ok=True)
    
    # Collection for visualization
    true_labels = []
    predicted_labels = []
    sample_tensors = []  # For correlation heatmap

    # Load watermark once
    try:
        watermark_img = safe_open_image(watermark_path)
        if watermark_img is None:
            print(f"Error: Could not load watermark from {watermark_path}")
            return 0, [], true_labels, predicted_labels
    except Exception as e:
        print(f"Error loading watermark: {str(e)}")
        return 0, [], true_labels, predicted_labels

    # No images to process
    if not image_files:
        print(f"No image files found to process")
        return 0, [], true_labels, predicted_labels

    # Determine processing mode based on available resources
    use_gpu_batch = torch.cuda.is_available() and not force_cpu
    
    results = []
    if use_gpu_batch:
        # Prepare watermark tensor once for batched processing
        # Determine resolution based on quality setting
        if quality == 'low':
            process_size = (64, 64)
        elif quality == 'medium':
            process_size = (128, 128)
        else:  # high
            process_size = (256, 256)
            
        transform = transforms.Compose([
            transforms.Resize(process_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        watermark = watermark_img.resize(process_size, Image.BILINEAR)
        watermark_tensor = transform(watermark).unsqueeze(0).to(device)
        
        # Calculate optimal batch size based on available GPU memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = available_memory - torch.cuda.memory_allocated()
        
        # Estimate memory per image based on resolution
        if quality == 'low':
            memory_per_image = 6 * 1024 * 1024  # ~6MB per image
        elif quality == 'medium':
            memory_per_image = 24 * 1024 * 1024  # ~24MB per image
        else:
            memory_per_image = 96 * 1024 * 1024  # ~96MB per image
            
        # Reserve 20% of memory for overhead
        safe_memory = free_memory * 0.8
        batch_size = min(max(1, int(safe_memory // memory_per_image)), 32)  # Cap at 32
        
        print(f"Processing with GPU batches of size {batch_size}, quality={quality}")
        
        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+min(batch_size, len(image_files)-i)]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch_files)} files)")
            
            batch_results, batch_true_labels, batch_pred_labels = process_image_batch(
                batch_files,
                input_dir,
                watermark_tensor,
                output_unwatermarked,
                output_watermarked,
                model,
                quality
            )
            
            results.extend(batch_results)
            true_labels.extend(batch_true_labels)
            predicted_labels.extend(batch_pred_labels)
            
            # Collect sample tensors for correlation heatmap (limit to avoid memory issues)
            if create_visualizations and len(sample_tensors) < 100:
                for filename in batch_files[:min(5, len(batch_files))]:  # Take at most 5 from each batch
                    try:
                        img_path = os.path.join(input_dir, filename)
                        image = safe_open_image(img_path)
                        if image is not None:
                            image_tensor = transform(image).unsqueeze(0)
                            sample_tensors.append(image_tensor)
                    except Exception:
                        continue
            
            # Clean up memory after each batch
            torch.cuda.empty_cache()
    else:
        # Fall back to parallel CPU processing
        num_workers = min(os.cpu_count() or 1, 4)  # Cap at 4 workers
        print(f"Processing with CPU using {num_workers} workers, quality={quality}")
        
        # Modified to collect true/predicted labels
        for filename in tqdm(image_files, desc="Processing images"):
            # Estimate true label from filename (heuristic)
            true_has_watermark = 'watermark' in filename.lower()
            true_labels.append(1 if true_has_watermark else 0)
            
            # Process image
            result = optimized_process_single_image(
                filename, 
                input_dir=input_dir,
                watermark_img=watermark_img,
                output_unwatermarked=output_unwatermarked,
                output_watermarked=output_watermarked,
                model=model,
                quality=quality
            )
            
            # Determine predicted label based on result
            if "Added watermark" in result:
                predicted_labels.append(0)  # Predicted clean, added watermark
            elif "Removed watermark" in result:
                predicted_labels.append(1)  # Predicted watermarked, removed watermark
            elif "Skipped" in result:
                # For skipped images, use a heuristic based on the confidence
                confidence_str = result.split("confidence: ")[1].split(")")[0]
                confidence = float(confidence_str)
                predicted_labels.append(1 if confidence > 0.5 else 0)
            else:
                # Error case
                predicted_labels.append(1 if true_has_watermark else 0)  # Default to true label
            
            results.append(result)
            
            # Collect sample tensors for correlation heatmap (limit to avoid memory issues)
            if create_visualizations and len(sample_tensors) < 50:
                try:
                    img_path = os.path.join(input_dir, filename)
                    image = safe_open_image(img_path)
                    if image is not None:
                        image_tensor = transform(image).unsqueeze(0)
                        sample_tensors.append(image_tensor)
                except Exception:
                    continue

    # Create visualizations if requested
    if create_visualizations and len(true_labels) > 0 and len(predicted_labels) > 0:
        # Create confusion matrix
        create_confusion_matrix(true_labels, predicted_labels)
        
        # Create correlation heatmap if we have enough samples
        if len(sample_tensors) > 10:
            # Only use a subset to avoid memory issues
            sample_tensors = sample_tensors[:min(50, len(sample_tensors))]
            sample_batch = torch.cat(sample_tensors, dim=0).to(device)
            create_correlation_heatmap(model, sample_batch)

    # Count processed images
    processed_count = sum(1 for result in results if not result.startswith("Skipped") and not result.startswith("Error"))
    
    return processed_count, results, true_labels, predicted_labels

# Context manager for null context when not using autocast
class nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): pass

# Main function with error handling and user options
def optimized_main():
    try:
        # Create visualizations directory
        os.makedirs("visualizations", exist_ok=True)
        
        # Define data directories
        train_clean_dir = "wm-nowm/train/no-watermark"
        train_watermarked_dir = "wm-nowm/train/watermark"
        val_clean_dir = "wm-nowm/valid/no-watermark"
        val_watermarked_dir = "wm-nowm/valid/watermark" 
        
        # Check if directories exist
        for directory in [train_clean_dir, train_watermarked_dir, val_clean_dir, val_watermarked_dir]:
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} does not exist.")
        
        # Allow user to choose whether to train or just process images
        mode = input("Choose mode (1: Train + Process, 2: Process only): ").strip()
        
        if mode == "2":
            print("Processing mode only - will load existing model if available.")
            should_train = False
        else:
            print("Training + Processing mode selected.")
            should_train = True
            
        # Ask about quality settings
        quality_choice = input("Choose processing quality (1: Low, 2: Medium, 3: High): ").strip()
        if quality_choice == "1":
            quality = "low"
            print("Using low quality (faster but less accurate).")
        elif quality_choice == "3":
            quality = "high"
            print("Using high quality (slower but more accurate).")
        else:
            quality = "medium"
            print("Using medium quality (balanced).")
        
        # Model parameters based on quality
        if quality == "low":
            process_size = (64, 64)
        elif quality == "medium":
            process_size = (128, 128)
        else:  # high
            process_size = (256, 256)
            
        # Get watermark path from user for training
        if should_train:
            watermark_path = input("Enter path to watermark image for training (or press Enter to use a blank watermark): ")
            if not watermark_path or not os.path.exists(watermark_path):
                print("No valid watermark path provided. Will use a blank watermark during training.")
                watermark_path = None
        
            # Optimized transforms with appropriate image size
            transform = transforms.Compose([
                transforms.Resize(process_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Create datasets
            train_dataset = WatermarkDataset(
                clean_dir=train_clean_dir,
                watermarked_dir=train_watermarked_dir,
                watermark_path=watermark_path,
                transform=transform,
                is_train=True
            )
            
            val_dataset = WatermarkDataset(
                clean_dir=val_clean_dir, 
                watermarked_dir=val_watermarked_dir,
                watermark_path=watermark_path,
                transform=transform,
                is_train=False
            )
            
            # Create visualizations for dataset distributions
            visualize_dataset_distribution(train_dataset, val_dataset)
            
            # Optimize batch size based on system resources and quality
            gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
            
            if quality == "low":
                batch_size = 64 if gpu_available else 32
            elif quality == "medium":
                batch_size = 32 if gpu_available else 16
            else:  # high
                batch_size = 16 if gpu_available else 8
            
            # Use only 1 worker for DataLoader to avoid broken pipe errors
            num_workers_dl = 0
            
            # Use pin_memory for faster data transfer to GPU
            pin_memory = gpu_available
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers_dl,
                pin_memory=pin_memory,
                persistent_workers=False
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers_dl,
                pin_memory=pin_memory,
                persistent_workers=False
            )
        
        # Create or load model
        watermark_strength = 0.5 if quality == "low" else 0.8
        model = OptimizedHiDDeNModel(watermark_strength=watermark_strength)
        
        # Path for saved model
        model_path = "watermark_model.pth"
        best_model_path = "watermark_model_best.pth"
        
        # Check for existing model files
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        elif os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif not should_train:
            print("No model file found. Please run training first or choose an existing model.")
            return
            
        # Train model if requested
        if should_train:
            print("Training model...")
            try:
                # Enable cuDNN benchmarking for faster training if available
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                
                model = optimized_train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=10,
                    learning_rate=0.001,
                    visualize=True  # Enable training visualization
                )
                
                # Save the final model
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
                
                # Disable benchmarking for inference
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = False
            except Exception as e:
                print(f"Error during training: {e}")
                # If best model exists, load it
                if os.path.exists(best_model_path):
                    print(f"Loading last best model from {best_model_path}")
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                else:
                    print("Training failed and no best model found. Exiting.")
                    return
                
        # Process user-provided images
        print("\n--- Watermark Processing ---")
        input_dir = input("Enter the directory path containing images to process: ")
        
        if not input_dir or not os.path.exists(input_dir):
            print("Error: Invalid input directory.")
            return
        
        watermark_path = input("Enter the path to the watermark image you want to apply: ")
        if not watermark_path or not os.path.exists(watermark_path):
            print("Error: Invalid watermark path.")
            return
        
        output_unwatermarked = "output/unwatermarked"
        output_watermarked = "output/watermarked"
        
        # Get list of image files
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
        
        if len(image_files) == 0:
            print(f"No image files found in {input_dir}")
            return
            
        # Ask about chunk processing for large datasets
        if len(image_files) > 100:
            chunked = input(f"Found {len(image_files)} images. Process in chunks? (y/n): ").strip().lower()
            if chunked == 'y':
                print("Processing in chunks to manage memory...")
                chunk_size = 50  # Choose a reasonable chunk size
                total_processed, results, true_labels, pred_labels = process_in_chunks(
                    image_files, 
                    chunk_size=chunk_size,
                    input_dir=input_dir,
                    watermark_path=watermark_path,
                    output_unwatermarked=output_unwatermarked,
                    output_watermarked=output_watermarked,
                    model=model,
                    quality=quality,
                    create_visualizations=True
                )
            else:
                # Process all at once
                print(f"Processing all {len(image_files)} images at once...")
                total_processed, results, true_labels, pred_labels = optimized_process_images(
                    image_files,
                    input_dir=input_dir,
                    watermark_path=watermark_path,
                    output_unwatermarked=output_unwatermarked,
                    output_watermarked=output_watermarked,
                    model=model,
                    quality=quality,
                    create_visualizations=True
                )
        else:
            # Process all at once for smaller datasets
            print(f"Processing {len(image_files)} images...")
            total_processed, results, true_labels, pred_labels = optimized_process_images(
                image_files,
                input_dir=input_dir,
                watermark_path=watermark_path,
                output_unwatermarked=output_unwatermarked,
                output_watermarked=output_watermarked,
                model=model,
                quality=quality,
                create_visualizations=True
            )
        
        # Create confusion matrix if we have enough data
        if len(true_labels) > 0 and len(pred_labels) > 0:
            create_confusion_matrix(true_labels, pred_labels)
        
        # Print summary results
        skipped = sum(1 for r in results if r.startswith("Skipped"))
        errors = sum(1 for r in results if r.startswith("Error"))
        watermarked = sum(1 for r in results if "Added watermark" in r)
        unwatermarked = sum(1 for r in results if "Removed watermark" in r)
        
        print("\n----- Processing Summary -----")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully processed: {total_processed}")
        print(f"  - Watermarked: {watermarked}")
        print(f"  - Unwatermarked: {unwatermarked}")
        print(f"Skipped (low confidence): {skipped}")
        print(f"Errors: {errors}")
        print(f"Unwatermarked images saved to: {output_unwatermarked}")
        print(f"Watermarked images saved to: {output_watermarked}")
        print(f"Visualizations saved to: visualizations/")
        
        # Ask if user wants to see detailed results
        if input("Show detailed results? (y/n): ").strip().lower() == 'y':
            for result in results:
                print(result)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# Main entry point
if __name__ == "__main__":
    optimized_main()