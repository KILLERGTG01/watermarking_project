import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import sys

# Output directories
EMBED_OUTPUT_DIR = 'output/stegnographed'
DECODE_OUTPUT_DIR = 'output/stegnograph_decoded'

# Device selection: CUDA → MPS → CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"🖥️ Using device: {device}")

def message_to_bits(message):
    """Convert a string message to a binary string"""
    return ''.join(format(ord(char), '08b') for char in message)

def bits_to_message(bits):
    """Convert a binary string back to a string message"""
    # Ensure bit length is multiple of 8
    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]
    
    message = ""
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):  # Make sure we have a full byte
            byte = bits[i:i+8]
            message += chr(int(byte, 2))
    return message

def create_pixel_map(img_shape):
    """Create a simple deterministic pixel map for embedding/extraction"""
    # Create a 1D array with all pixel indices
    total_pixels = img_shape[0] * img_shape[1]
    
    # Create a simple pattern - can be made more sophisticated
    # Currently using a simple sequential pattern
    return np.arange(total_pixels)

def embed_message(img_np, message):
    """Embed a message into an image using LSB steganography"""
    # Make a copy to avoid modifying the original
    stego_img = img_np.copy()
    
    # Get total available pixels
    h, w, _ = stego_img.shape
    total_pixels = h * w
    
    # Convert message to binary
    binary_message = message_to_bits(message)
    
    # Create 32-bit header with message length
    header = format(len(binary_message), '032b')
    final_bits = header + binary_message
    
    # Check if image is large enough to hold the message
    if len(final_bits) > total_pixels:
        print(f"Message too long ({len(final_bits)} bits) for image ({total_pixels} pixels)")
        return None
    
    # Get pixel map
    pixel_map = create_pixel_map((h, w))
    
    # Embed each bit
    for i, bit in enumerate(final_bits):
        if i >= len(pixel_map):
            break
            
        # Get pixel coordinates
        idx = pixel_map[i]
        y, x = idx // w, idx % w
        
        # Modify least significant bit of blue channel
        stego_img[y, x, 2] = (stego_img[y, x, 2] & 0xFE) | int(bit)
    
    return stego_img

def extract_message(img_np):
    """Extract a message from an image using LSB steganography"""
    h, w, _ = img_np.shape
    total_pixels = h * w
    
    # Get pixel map (same as used for embedding)
    pixel_map = create_pixel_map((h, w))
    
    # Extract header bits (first 32 bits)
    header_bits = ""
    for i in range(32):
        if i >= len(pixel_map):
            break
            
        idx = pixel_map[i]
        y, x = idx // w, idx % w
        header_bits += str(img_np[y, x, 2] & 1)
    
    # Parse message length from header
    try:
        message_len = int(header_bits, 2)
        
        # Check for invalid message length
        if message_len <= 0 or message_len > total_pixels - 32:
            raise ValueError(f"Invalid message length detected: {message_len}")
    except Exception as e:
        raise ValueError(f"Failed to parse header: {str(e)}")
    
    # Extract message bits
    message_bits = ""
    for i in range(32, 32 + message_len):
        if i >= len(pixel_map):
            break
            
        idx = pixel_map[i]
        y, x = idx // w, idx % w
        message_bits += str(img_np[y, x, 2] & 1)
    
    # Convert bits back to message
    return bits_to_message(message_bits)

def is_supported_image(file_path):
    """Check if a file has a supported image extension"""
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    return os.path.isfile(file_path) and file_path.lower().endswith(supported_ext)

def process_single_image_embed(input_path, output_path, message):
    """Process a single image for embedding a message"""
    try:
        # Open and convert to RGB
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)
        
        # Embed message
        stego_np = embed_message(img_np, message)
        
        if stego_np is None:
            print(f"❌ Failed to embed: Image too small for the message")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Force PNG format to avoid any lossy compression
        output_path = os.path.splitext(output_path)[0] + '.png'
        
        # Save the image
        Image.fromarray(stego_np.astype(np.uint8)).save(output_path, format='PNG')
        print(f"✅ Successfully embedded message in: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error embedding message: {str(e)}")
        return False

def process_single_image_decode(input_path):
    """Process a single image for decoding a message"""
    try:
        # Open and convert to RGB
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)
        
        # Extract message
        message = extract_message(img_np)
        
        print(f"📝 Decoded message: {message}")
        return message
    except Exception as e:
        print(f"❌ Error decoding message: {str(e)}")
        return None

def batch_embed(input_folder, message):
    """Embed the same message into multiple images"""
    os.makedirs(EMBED_OUTPUT_DIR, exist_ok=True)
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    
    # Support both file and directory inputs
    if os.path.isfile(input_folder) and input_folder.lower().endswith(supported_ext):
        files = [os.path.basename(input_folder)]
        input_folder = os.path.dirname(input_folder)
        if not input_folder:  # Handle case where only filename is provided
            input_folder = '.'
    else:
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_ext)]

    total, success, failed = 0, 0, 0

    print(f"\n🚀 Embedding message into {len(files)} images...")
    for filename in tqdm(files, desc="Embedding"):
        total += 1
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(EMBED_OUTPUT_DIR, filename)
        
        try:
            if process_single_image_embed(input_path, output_path, message):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            failed += 1

    print("\n📊 Embed Summary")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {EMBED_OUTPUT_DIR}")

def batch_decode(input_folder):
    """Decode messages from multiple images"""
    os.makedirs(DECODE_OUTPUT_DIR, exist_ok=True)
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    
    # Support both file and directory inputs
    if os.path.isfile(input_folder) and input_folder.lower().endswith(supported_ext):
        files = [os.path.basename(input_folder)]
        input_folder = os.path.dirname(input_folder)
        if not input_folder:  # Handle case where only filename is provided
            input_folder = '.'
    else:
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_ext)]

    total, success, failed = 0, 0, 0

    print(f"\n🔍 Decoding messages from {len(files)} images...")
    for filename in tqdm(files, desc="Decoding"):
        total += 1
        input_path = os.path.join(input_folder, filename)
        
        try:
            message = process_single_image_decode(input_path)
            if message:
                out_txt = os.path.join(DECODE_OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.txt")
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write(message)
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            failed += 1

    print("\n📊 Decode Summary")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {DECODE_OUTPUT_DIR}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="GPU Steganography Tool")
    parser.add_argument("--mode", "-m", choices=["e", "d"], help="Mode: (e)mbed or (d)ecode")
    parser.add_argument("--input", "-i", help="Input image file or folder path")
    parser.add_argument("--output", "-o", help="Output file or folder path (for embed mode)")
    parser.add_argument("--message", help="Message to embed (for embed mode)")
    return parser.parse_args()

if __name__ == "__main__":
    print("🧠 Steganography Tool")
    
    args = parse_arguments()
    
    # If arguments are provided, use them
    if args.mode:
        mode = args.mode
        
        if mode == 'e':
            if not args.input:
                print("❌ Error: Input file or folder path is required for embed mode")
                sys.exit(1)
            
            input_path = args.input
            
            if not args.message:
                message = input("Enter the message to embed (emojis supported): ").strip()
            else:
                message = args.message
                
            # Handle single file
            if os.path.isfile(input_path):
                output_path = args.output if args.output else os.path.join(
                    EMBED_OUTPUT_DIR, 
                    os.path.basename(input_path)
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if process_single_image_embed(input_path, output_path, message):
                    print(f"✅ Message embedded successfully in: {output_path}")
                else:
                    print("❌ Failed to embed message")
            # Handle directory
            else:
                batch_embed(input_path, message)
                
        elif mode == 'd':
            if not args.input:
                print("❌ Error: Input file or folder path is required for decode mode")
                sys.exit(1)
                
            input_path = args.input
            
            # Handle single file
            if os.path.isfile(input_path):
                message = process_single_image_decode(input_path)
                if message and args.output:
                    os.makedirs(os.path.dirname(args.output), exist_ok=True)
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(message)
                    print(f"✅ Message saved to: {args.output}")
            # Handle directory
            else:
                batch_decode(input_path)
        else:
            print("❌ Invalid option. Please enter 'e' or 'd'.")
    # Interactive mode if no arguments provided
    else:
        mode = input("Do you want to (e)mbed or (d)ecode? ").strip().lower()
        
        if mode == 'e':
            input_path = input("Enter image file or folder path containing original images: ").strip()
            message = input("Enter the message to embed (emojis supported): ").strip()
            
            # Single file handling
            if os.path.isfile(input_path):
                output_dir = input("Enter output file path (or press Enter for default): ").strip()
                if not output_dir:
                    output_dir = os.path.join(EMBED_OUTPUT_DIR, os.path.basename(input_path))
                    
                if process_single_image_embed(input_path, output_dir, message):
                    print(f"✅ Message embedded successfully in: {output_dir}")
                else:
                    print("❌ Failed to embed message")
            else:
                batch_embed(input_path, message)
                
        elif mode == 'd':
            input_path = input("Enter image file or folder path containing stego images: ").strip()
            
            # Single file handling
            if os.path.isfile(input_path):
                message = process_single_image_decode(input_path)
                save_option = input("Save message to file? (y/n): ").strip().lower()
                if save_option == 'y' and message:
                    output_file = input("Enter output file path (or press Enter for default): ").strip()
                    if not output_file:
                        base_name = os.path.splitext(os.path.basename(input_path))[0]
                        output_file = os.path.join(DECODE_OUTPUT_DIR, f"{base_name}.txt")
                    
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(message)
                    print(f"✅ Message saved to: {output_file}")
            else:
                batch_decode(input_path)
        else:
            print("❌ Invalid option. Please enter 'e' or 'd'.")