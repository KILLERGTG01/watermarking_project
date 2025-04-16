import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import sys
import json
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Output directories
EMBED_OUTPUT_DIR = 'stegnograph/output/stegnographed'
DECODE_OUTPUT_DIR = 'stegnograph/output/stegnograph_decoded'
RESULTS_DIR = 'stegnograph/output/results'
CHARTS_DIR = 'stegnograph/output/charts'

# Ensure all directories exist
for directory in [EMBED_OUTPUT_DIR, DECODE_OUTPUT_DIR, RESULTS_DIR, CHARTS_DIR]:
    os.makedirs(directory, exist_ok=True)

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

def calculate_psnr(original, stego):
    """Calculate Peak Signal-to-Noise Ratio between original and stego images"""
    mse = np.mean((original.astype(float) - stego.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def calculate_snr(original, stego):
    """Calculate Signal-to-Noise Ratio between original and stego images"""
    signal_power = np.mean(original.astype(float) ** 2)
    noise_power = np.mean((original.astype(float) - stego.astype(float)) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * math.log10(signal_power / noise_power)
    return snr

def calculate_similarity(original, stego):
    """Calculate a simple similarity index (percentage of unchanged pixels)"""
    total_pixels = original.size
    different_pixels = np.sum(original != stego)
    similarity = (1 - different_pixels / total_pixels) * 100
    return similarity

def embed_message(img_np, message):
    """Embed a message into an image using LSB steganography"""
    # Start timing
    start_time = time.time()
    
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
        return None, None
    
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
    
    # End timing
    end_time = time.time()
    
    # Collect metrics
    metrics = {
        'embed_time': end_time - start_time,
        'message_size_bits': len(binary_message),
        'message_size_bytes': len(message),
        'header_size_bits': len(header),
        'total_bits_embedded': len(final_bits),
        'image_capacity_bits': total_pixels,
        'usage_percentage': (len(final_bits) / total_pixels) * 100,
        'psnr': calculate_psnr(img_np, stego_img),
        'snr': calculate_snr(img_np, stego_img),
        'similarity': calculate_similarity(img_np, stego_img)
    }
    
    return stego_img, metrics

def extract_message(img_np):
    """Extract a message from an image using LSB steganography"""
    # Start timing
    start_time = time.time()
    
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
        end_time = time.time()
        return None, {'extract_time': end_time - start_time, 'error': str(e)}
    
    # Extract message bits
    message_bits = ""
    for i in range(32, 32 + message_len):
        if i >= len(pixel_map):
            break
            
        idx = pixel_map[i]
        y, x = idx // w, idx % w
        message_bits += str(img_np[y, x, 2] & 1)
    
    # Convert bits back to message
    message = bits_to_message(message_bits)
    
    # End timing
    end_time = time.time()
    
    # Collect metrics
    metrics = {
        'extract_time': end_time - start_time,
        'message_size_bits': message_len,
        'message_size_bytes': len(message),
        'extraction_success': True
    }
    
    return message, metrics

def is_supported_image(file_path):
    """Check if a file has a supported image extension"""
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    return os.path.isfile(file_path) and file_path.lower().endswith(supported_ext)

def save_results(metrics, operation_type):
    """Save metrics to CSV and JSON files"""
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{operation_type}_{timestamp}"
    csv_path = os.path.join(RESULTS_DIR, f"{base_filename}.csv")
    json_path = os.path.join(RESULTS_DIR, f"{base_filename}.json")
    
    # Save as JSON (individual file)
    with open(json_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=2)
    
    # Save as CSV
    if isinstance(metrics, list) and metrics:
        import csv
        with open(csv_path, 'w', newline='') as csv_file:
            # Get all possible field names from all results
            all_fields = set()
            for result in metrics:
                all_fields.update(result.keys())
            
            writer = csv.DictWriter(csv_file, fieldnames=sorted(list(all_fields)))
            writer.writeheader()
            for result in metrics:
                writer.writerow(result)
    elif isinstance(metrics, dict):
        import csv
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=sorted(list(metrics.keys())))
            writer.writeheader()
            writer.writerow(metrics)
    
    print(f"📊 Results saved to {RESULTS_DIR}")
    print(f"  - CSV: {os.path.basename(csv_path)}")
    print(f"  - JSON: {os.path.basename(json_path)}")
    
    return csv_path, json_path

def generate_charts(metrics, filename_prefix):
    """Generate charts for the metrics"""
    if not isinstance(metrics, list) or not metrics:
        return None
    
    # Create charts directory if it doesn't exist
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    # 1. PSNR vs. File chart
    if any('psnr' in m for m in metrics):
        plt.figure(figsize=(10, 6))
        files = [os.path.basename(m.get('input_file', f"file{i}")) for i, m in enumerate(metrics)]
        psnr_values = [m.get('psnr', 0) for m in metrics]
        
        plt.bar(range(len(files)), psnr_values, color='skyblue')
        plt.xlabel('Image Files')
        plt.ylabel('PSNR (dB)')
        plt.title('Peak Signal-to-Noise Ratio for Each Image')
        plt.xticks(range(len(files)), files, rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = os.path.join(CHARTS_DIR, f"{filename_prefix}_psnr_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"📈 Generated PSNR chart: {os.path.basename(chart_path)}")
    
    # 2. SNR vs. File chart
    if any('snr' in m for m in metrics):
        plt.figure(figsize=(10, 6))
        files = [os.path.basename(m.get('input_file', f"file{i}")) for i, m in enumerate(metrics)]
        snr_values = [m.get('snr', 0) for m in metrics]
        
        plt.bar(range(len(files)), snr_values, color='lightgreen')
        plt.xlabel('Image Files')
        plt.ylabel('SNR (dB)')
        plt.title('Signal-to-Noise Ratio for Each Image')
        plt.xticks(range(len(files)), files, rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = os.path.join(CHARTS_DIR, f"{filename_prefix}_snr_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"📈 Generated SNR chart: {os.path.basename(chart_path)}")
    
    # 3. Similarity vs. File chart
    if any('similarity' in m for m in metrics):
        plt.figure(figsize=(10, 6))
        files = [os.path.basename(m.get('input_file', f"file{i}")) for i, m in enumerate(metrics)]
        similarity_values = [m.get('similarity', 0) for m in metrics]
        
        plt.bar(range(len(files)), similarity_values, color='salmon')
        plt.xlabel('Image Files')
        plt.ylabel('Similarity (%)')
        plt.title('Image Similarity Index for Each Image')
        plt.xticks(range(len(files)), files, rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = os.path.join(CHARTS_DIR, f"{filename_prefix}_similarity_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"📈 Generated Similarity chart: {os.path.basename(chart_path)}")
    
    # 4. Combined metrics comparison
    if any('psnr' in m and 'snr' in m and 'similarity' in m for m in metrics):
        plt.figure(figsize=(12, 8))
        
        # Normalize values for comparison
        max_psnr = max(m.get('psnr', 0) for m in metrics) or 1
        max_snr = max(m.get('snr', 0) for m in metrics) or 1
        # Similarity is already in percentage scale
        
        files = [os.path.basename(m.get('input_file', f"file{i}")) for i, m in enumerate(metrics)]
        
        # Create positions for grouped bars
        x = np.arange(len(files))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot normalized metrics
        rects1 = ax.bar(x - width, [m.get('psnr', 0) / max_psnr * 100 for m in metrics], width, label='PSNR (normalized)')
        rects2 = ax.bar(x, [m.get('snr', 0) / max_snr * 100 for m in metrics], width, label='SNR (normalized)')
        rects3 = ax.bar(x + width, [m.get('similarity', 0) for m in metrics], width, label='Similarity (%)')
        
        ax.set_xlabel('Image Files')
        ax.set_ylabel('Value (%)')
        ax.set_title('Comparison of Metrics Across Images')
        ax.set_xticks(x)
        ax.set_xticklabels(files, rotation=45, ha='right')
        ax.legend()
        
        # Add a table with the actual values below the chart
        table_data = []
        for i, m in enumerate(metrics):
            psnr = m.get('psnr', 0)
            snr = m.get('snr', 0)
            similarity = m.get('similarity', 0)
            table_data.append([f"{psnr:.2f} dB", f"{snr:.2f} dB", f"{similarity:.2f}%"])
        
        plt.tight_layout()
        
        chart_path = os.path.join(CHARTS_DIR, f"{filename_prefix}_combined_metrics.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"📈 Generated combined metrics chart: {os.path.basename(chart_path)}")
        
        # Create a separate figure for the table
        fig, ax = plt.subplots(figsize=(10, len(metrics) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=table_data,
            rowLabels=files,
            colLabels=["PSNR", "SNR", "Similarity"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Metrics Table')
        
        table_path = os.path.join(CHARTS_DIR, f"{filename_prefix}_metrics_table.png")
        plt.savefig(table_path)
        plt.close()
        print(f"📈 Generated metrics table: {os.path.basename(table_path)}")
    
    return CHARTS_DIR

def process_single_image_embed(input_path, output_path, message):
    """Process a single image for embedding a message"""
    try:
        # Open and convert to RGB
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)
        
        # Record original image size
        img_size = os.path.getsize(input_path)
        img_dimensions = img.size
        
        # Embed message
        stego_np, metrics = embed_message(img_np, message)
        
        if stego_np is None:
            print(f"❌ Failed to embed: Image too small for the message")
            return False, None
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Force PNG format to avoid any lossy compression
        output_path = os.path.splitext(output_path)[0] + '.png'
        
        # Save the image
        Image.fromarray(stego_np.astype(np.uint8)).save(output_path, format='PNG')
        
        # Record stego image size
        stego_size = os.path.getsize(output_path)
        
        # Add more metrics
        metrics.update({
            'original_image_size_bytes': img_size,
            'stego_image_size_bytes': stego_size,
            'size_increase_bytes': stego_size - img_size,
            'size_increase_percentage': ((stego_size - img_size) / img_size) * 100 if img_size > 0 else 0,
            'image_width': img_dimensions[0],
            'image_height': img_dimensions[1],
            'input_file': input_path,
            'output_file': output_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        print(f"✅ Successfully embedded message in: {output_path}")
        return True, metrics
    except Exception as e:
        print(f"❌ Error embedding message: {str(e)}")
        return False, {'error': str(e), 'input_file': input_path}

def process_single_image_decode(input_path):
    """Process a single image for decoding a message"""
    try:
        # Open and convert to RGB
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)
        
        # Record image info
        img_size = os.path.getsize(input_path)
        img_dimensions = img.size
        
        # Extract message
        message, metrics = extract_message(img_np)
        
        if message is None:
            print(f"❌ Error decoding message: {metrics.get('error', 'Unknown error')}")
            return None, metrics
        
        # Add more metrics
        metrics.update({
            'image_size_bytes': img_size,
            'image_width': img_dimensions[0],
            'image_height': img_dimensions[1],
            'input_file': input_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        print(f"📝 Decoded message: {message}")
        return message, metrics
    except Exception as e:
        print(f"❌ Error decoding message: {str(e)}")
        return None, {'error': str(e), 'input_file': input_path}

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
    all_metrics = []

    print(f"\n🚀 Embedding message into {len(files)} images...")
    for filename in tqdm(files, desc="Embedding"):
        total += 1
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(EMBED_OUTPUT_DIR, filename)
        
        try:
            result, metrics = process_single_image_embed(input_path, output_path, message)
            if result:
                success += 1
                if metrics:
                    all_metrics.append(metrics)
            else:
                failed += 1
                if metrics:
                    all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            failed += 1
            all_metrics.append({
                'input_file': input_path, 
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Save results
    csv_path, json_path = save_results(all_metrics, "embed_batch")
    
    # Generate charts if we have successful embeddings
    if any('psnr' in m for m in all_metrics):
        charts_dir = generate_charts(all_metrics, "embed_batch")

    print("\n📊 Embed Summary")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {EMBED_OUTPUT_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Charts:  {CHARTS_DIR}")
    
    # If metrics available, show averages
    if all_metrics:
        successful_metrics = [m for m in all_metrics if 'error' not in m]
        if successful_metrics:
            avg_psnr = sum(m.get('psnr', 0) for m in successful_metrics) / len(successful_metrics)
            avg_snr = sum(m.get('snr', 0) for m in successful_metrics) / len(successful_metrics)
            avg_similarity = sum(m.get('similarity', 0) for m in successful_metrics) / len(successful_metrics)
            avg_time = sum(m.get('embed_time', 0) for m in successful_metrics) / len(successful_metrics)
            
            print("\n📈 Average Metrics:")
            print(f"  PSNR:       {avg_psnr:.2f} dB")
            print(f"  SNR:        {avg_snr:.2f} dB")
            print(f"  Similarity: {avg_similarity:.2f}%")
            print(f"  Time:       {avg_time:.4f} seconds")
    
    return all_metrics

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
    all_metrics = []

    print(f"\n🔍 Decoding messages from {len(files)} images...")
    for filename in tqdm(files, desc="Decoding"):
        total += 1
        input_path = os.path.join(input_folder, filename)
        
        try:
            message, metrics = process_single_image_decode(input_path)
            if message:
                out_txt = os.path.join(DECODE_OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.txt")
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write(message)
                success += 1
                if metrics:
                    all_metrics.append(metrics)
            else:
                failed += 1
                if metrics:
                    all_metrics.append(metrics)
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            failed += 1
            all_metrics.append({
                'input_file': input_path, 
                'error': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Save results
    csv_path, json_path = save_results(all_metrics, "decode_batch")

    print("\n📊 Decode Summary")
    print(f"  Total:   {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {DECODE_OUTPUT_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    # If metrics available, show averages
    if all_metrics:
        successful_metrics = [m for m in all_metrics if m.get('extraction_success', False)]
        if successful_metrics:
            avg_time = sum(m.get('extract_time', 0) for m in successful_metrics) / len(successful_metrics)
            avg_size = sum(m.get('message_size_bytes', 0) for m in successful_metrics) / len(successful_metrics)
            
            print("\n📈 Average Metrics:")
            print(f"  Extraction Time: {avg_time:.4f} seconds")
            print(f"  Message Size:    {avg_size:.2f} bytes")
    
    return all_metrics

def generate_full_report(embed_metrics=None, decode_metrics=None):
    """Generate a comprehensive HTML report from metrics"""
    if not embed_metrics and not decode_metrics:
        print("❌ No metrics available for report generation")
        return None
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(RESULTS_DIR, f"steganography_report_{timestamp}.html")
    
    # Basic HTML template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Steganography Results Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .chart-container {{ display: flex; justify-content: space-between; margin: 20px 0; }}
            .chart {{ flex: 1; margin: 0 10px; height: 300px; background-color: #f9f9f9; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Steganography Results Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Summary</h2>
    """
    
    # Add embedding summary if available
    if embed_metrics:
        successful_embeds = [m for m in embed_metrics if 'error' not in m]
        html += f"""
            <h3>Embedding</h3>
            <p>Total images processed: {len(embed_metrics)}</p>
            <p>Successful embeddings: {len(successful_embeds)}</p>
            <p>Failed embeddings: {len(embed_metrics) - len(successful_embeds)}</p>
        """
        
        if successful_embeds:
            avg_psnr = sum(m.get('psnr', 0) for m in successful_embeds) / len(successful_embeds)
            avg_snr = sum(m.get('snr', 0) for m in successful_embeds) / len(successful_embeds)
            avg_similarity = sum(m.get('similarity', 0) for m in successful_embeds) / len(successful_embeds)
            avg_time = sum(m.get('embed_time', 0) for m in successful_embeds) / len(successful_embeds)
            
            html += f"""
            <p>Average PSNR: {avg_psnr:.2f} dB</p>
            <p>Average SNR: {avg_snr:.2f} dB</p>
            <p>Average Similarity: {avg_similarity:.2f}%</p>
            <p>Average Embedding Time: {avg_time:.4f} seconds</p>
            """
    
    # Add decoding summary if available
    if decode_metrics:
        successful_decodes = [m for m in decode_metrics if m.get('extraction_success', False)]
        html += f"""
            <h3>Decoding</h3>
            <p>Total images processed: {len(decode_metrics)}</p>
            <p>Successful decodings: {len(successful_decodes)}</p>
            <p>Failed decodings: {len(decode_metrics) - len(successful_decodes)}</p>
        """
        
        if successful_decodes:
            avg_time = sum(m.get('extract_time', 0) for m in successful_decodes) / len(successful_decodes)
            avg_size = sum(m.get('message_size_bytes', 0) for m in successful_decodes) / len(successful_decodes)
            
            html += f"""
            <p>Average Extraction Time: {avg_time:.4f} seconds</p>
            <p>Average Message Size: {avg_size:.2f} bytes</p>
            """
    
    html += """
        </div>
    """
    
    # Add embedding details if available
    if embed_metrics:
        html += """
        <h2>Embedding Details</h2>
        <table>
            <tr>
                <th>File</th>
                <th>PSNR (dB)</th>
                <th>SNR (dB)</th>
                <th>Similarity (%)</th>
                <th>Time (s)</th>
                <th>Message Size (bytes)</th>
                <th>Usage (%)</th>
            </tr>
        """
        
        for metric in embed_metrics:
            if 'error' not in metric:
                # Fix for conditional f-string formatting
                psnr_val = metric.get('psnr', 'N/A')
                psnr_str = f"{psnr_val:.2f}" if isinstance(psnr_val, (int, float)) else 'N/A'
                
                snr_val = metric.get('snr', 'N/A')
                snr_str = f"{snr_val:.2f}" if isinstance(snr_val, (int, float)) else 'N/A'
                
                sim_val = metric.get('similarity', 'N/A')
                sim_str = f"{sim_val:.2f}" if isinstance(sim_val, (int, float)) else 'N/A'
                
                time_val = metric.get('embed_time', 'N/A')
                time_str = f"{time_val:.4f}" if isinstance(time_val, (int, float)) else 'N/A'
                
                usage_val = metric.get('usage_percentage', 'N/A')
                usage_str = f"{usage_val:.2f}" if isinstance(usage_val, (int, float)) else 'N/A'
                
                html += f"""
                <tr>
                    <td>{os.path.basename(metric.get('input_file', 'unknown'))}</td>
                    <td>{psnr_str}</td>
                    <td>{snr_str}</td>
                    <td>{sim_str}</td>
                    <td>{time_str}</td>
                    <td>{metric.get('message_size_bytes', 'N/A')}</td>
                    <td>{usage_str}</td>
                </tr>
                """
        
        html += """
        </table>
        """
    
    # Add decoding details if available
    if decode_metrics:
        html += """
        <h2>Decoding Details</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Extraction Time (s)</th>
                <th>Message Size (bytes)</th>
                <th>Status</th>
            </tr>
        """
        
        for metric in decode_metrics:
            status = "Success" if metric.get('extraction_success', False) else f"Failed: {metric.get('error', 'Unknown error')}"
            
            # Fix for conditional f-string formatting
            time_val = metric.get('extract_time', 'N/A')
            time_str = f"{time_val:.4f}" if isinstance(time_val, (int, float)) else 'N/A'
            
            html += f"""
            <tr>
                <td>{os.path.basename(metric.get('input_file', 'unknown'))}</td>
                <td>{time_str}</td>
                <td>{metric.get('message_size_bytes', 'N/A')}</td>
                <td>{status}</td>
            </tr>
            """
        
        html += """
        </table>
        """
    
    # Reference charts if they exist
    html += """
        <h2>Visualization Charts</h2>
        <p>Charts are available in the charts directory:</p>
        <ul>
    """
    
    for file in os.listdir(CHARTS_DIR):
        if file.endswith('.png'):
            html += f"""
            <li>{file}</li>
            """
    
    html += """
        </ul>
    """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"📑 Full report generated: {html_path}")
    return html_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Steganography Tool with Metrics")
    parser.add_argument("--mode", "-m", choices=["e", "d", "r"], help="Mode: (e)mbed, (d)ecode, or generate (r)eport")
    parser.add_argument("--input", "-i", help="Input image file or folder path")
    parser.add_argument("--output", "-o", help="Output file or folder path (for embed mode)")
    parser.add_argument("--message", help="Message to embed (for embed mode)")
    parser.add_argument("--report", "-r", action="store_true", help="Generate detailed report after operation")
    parser.add_argument("--embed-results", help="Path to embedding results JSON for report generation")
    parser.add_argument("--decode-results", help="Path to decoding results JSON for report generation")
    parser.add_argument("--visualization", "-v", action="store_true", help="Generate visualizations")
    return parser.parse_args()

if __name__ == "__main__":
    print("🧠 Steganography Tool with Metrics and Visualization")
    
    args = parse_arguments()
    
    # If arguments are provided, use them
    if args.mode:
        mode = args.mode
        
        # Handle report generation mode
        if mode == 'r':
            embed_data = None
            decode_data = None
            
            if args.embed_results and os.path.isfile(args.embed_results):
                with open(args.embed_results, 'r') as f:
                    embed_data = json.load(f)
            
            if args.decode_results and os.path.isfile(args.decode_results):
                with open(args.decode_results, 'r') as f:
                    decode_data = json.load(f)
            
            generate_full_report(embed_data, decode_data)
            
            # Generate visualizations if requested
            if args.visualization and (embed_data or decode_data):
                if embed_data:
                    generate_charts(embed_data, "report_embed")
                if decode_data:
                    # Only generate time-based charts for decode data
                    # as decode doesn't have PSNR, SNR, similarity
                    plt.figure(figsize=(10, 6))
                    files = [os.path.basename(m.get('input_file', f"file{i}")) for i, m in enumerate(decode_data) if m.get('extraction_success', False)]
                    times = [m.get('extract_time', 0) for m in decode_data if m.get('extraction_success', False)]
                    
                    if files and times:
                        plt.bar(range(len(files)), times, color='lightblue')
                        plt.xlabel('Image Files')
                        plt.ylabel('Extraction Time (s)')
                        plt.title('Message Extraction Time for Each Image')
                        plt.xticks(range(len(files)), files, rotation=45, ha='right')
                        plt.tight_layout()
                        
                        chart_path = os.path.join(CHARTS_DIR, f"report_decode_time_chart.png")
                        plt.savefig(chart_path)
                        plt.close()
                        print(f"📈 Generated extraction time chart: {os.path.basename(chart_path)}")
            
            sys.exit(0)
        
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
                
                success, metrics = process_single_image_embed(input_path, output_path, message)
                if success:
                    print(f"✅ Message embedded successfully in: {output_path}")
                    
                    # Save single result
                    if metrics:
                        csv_path, json_path = save_results([metrics], "embed_single")
                    
                    # Generate report if requested
                    if args.report and metrics:
                        generate_full_report([metrics], None)
                    
                    # Generate visualization if requested
                    if args.visualization and metrics:
                        generate_charts([metrics], "embed_single")
                else:
                    print("❌ Failed to embed message")
            # Handle directory
            else:
                metrics = batch_embed(input_path, message)
                
                # Generate report if requested
                if args.report and metrics:
                    generate_full_report(metrics, None)
                
        elif mode == 'd':
            if not args.input:
                print("❌ Error: Input file or folder path is required for decode mode")
                sys.exit(1)
                
            input_path = args.input
            
            # Handle single file
            if os.path.isfile(input_path):
                message, metrics = process_single_image_decode(input_path)
                if message:
                    if args.output:
                        os.makedirs(os.path.dirname(args.output), exist_ok=True)
                        with open(args.output, 'w', encoding='utf-8') as f:
                            f.write(message)
                        print(f"✅ Message saved to: {args.output}")
                    
                    # Save single result
                    if metrics:
                        csv_path, json_path = save_results([metrics], "decode_single")
                    
                    # Generate report if requested
                    if args.report and metrics:
                        generate_full_report(None, [metrics])
            # Handle directory
            else:
                metrics = batch_decode(input_path)
                
                # Generate report if requested
                if args.report and metrics:
                    generate_full_report(None, metrics)
        else:
            print("❌ Invalid option. Please enter 'e' for embed, 'd' for decode, or 'r' for report generation")
            sys.exit(1)
    # Interactive mode if no arguments provided
    else:
        while True:
            print("\n🔧 Steganography Tool Options:")
            print("  1. Embed message in image(s)")
            print("  2. Extract message from image(s)")
            print("  3. Generate report from previous results")
            print("  4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                input_path = input("Enter input image file or folder path: ").strip()
                if not os.path.exists(input_path):
                    print(f"❌ Path does not exist: {input_path}")
                    continue
                    
                message = input("Enter the message to embed (emojis supported): ").strip()
                
                if os.path.isfile(input_path):
                    output_path = input("Enter output image path (leave blank for default): ").strip()
                    if not output_path:
                        output_path = os.path.join(EMBED_OUTPUT_DIR, os.path.basename(input_path))
                    
                    success, metrics = process_single_image_embed(input_path, output_path, message)
                    if success:
                        print(f"✅ Message embedded successfully in: {output_path}")
                        
                        # Save single result
                        if metrics:
                            csv_path, json_path = save_results([metrics], "embed_single")
                        
                        # Ask about report and visualization
                        if input("Generate detailed report? (y/n): ").lower().startswith('y'):
                            generate_full_report([metrics], None)
                        
                        if input("Generate visualizations? (y/n): ").lower().startswith('y'):
                            generate_charts([metrics], "embed_single")
                    else:
                        print("❌ Failed to embed message")
                else:
                    metrics = batch_embed(input_path, message)
                    
                    # Ask about report
                    if input("Generate detailed report? (y/n): ").lower().startswith('y'):
                        generate_full_report(metrics, None)
                
            elif choice == '2':
                input_path = input("Enter input image file or folder path: ").strip()
                if not os.path.exists(input_path):
                    print(f"❌ Path does not exist: {input_path}")
                    continue
                
                if os.path.isfile(input_path):
                    message, metrics = process_single_image_decode(input_path)
                    if message:
                        output_path = input("Enter output text file path (leave blank for console only): ").strip()
                        if output_path:
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(message)
                            print(f"✅ Message saved to: {output_path}")
                        
                        # Save single result
                        if metrics:
                            csv_path, json_path = save_results([metrics], "decode_single")
                        
                        # Ask about report
                        if input("Generate detailed report? (y/n): ").lower().startswith('y'):
                            generate_full_report(None, [metrics])
                else:
                    metrics = batch_decode(input_path)
                    
                    # Ask about report
                    if input("Generate detailed report? (y/n): ").lower().startswith('y'):
                        generate_full_report(None, metrics)
            
            elif choice == '3':
                print("\nReport Generation:")
                embed_json = input("Enter path to embedding results JSON (leave blank if none): ").strip()
                decode_json = input("Enter path to decoding results JSON (leave blank if none): ").strip()
                
                embed_data = None
                decode_data = None
                
                if embed_json and os.path.isfile(embed_json):
                    with open(embed_json, 'r') as f:
                        try:
                            embed_data = json.load(f)
                            print(f"✅ Loaded embedding results from: {embed_json}")
                        except json.JSONDecodeError:
                            print(f"❌ Invalid JSON file: {embed_json}")
                
                if decode_json and os.path.isfile(decode_json):
                    with open(decode_json, 'r') as f:
                        try:
                            decode_data = json.load(f)
                            print(f"✅ Loaded decoding results from: {decode_json}")
                        except json.JSONDecodeError:
                            print(f"❌ Invalid JSON file: {decode_json}")
                
                if embed_data or decode_data:
                    generate_full_report(embed_data, decode_data)
                    
                    # Ask about visualizations
                    if input("Generate visualizations? (y/n): ").lower().startswith('y'):
                        if embed_data:
                            generate_charts(embed_data, "report_embed")
                        if decode_data and any('extract_time' in m for m in decode_data):
                            # Only time-based charts make sense for decode data
                            plt.figure(figsize=(10, 6))
                            files = [os.path.basename(m.get('input_file', f"file{i}")) for i, m in enumerate(decode_data) if m.get('extraction_success', False)]
                            times = [m.get('extract_time', 0) for m in decode_data if m.get('extraction_success', False)]
                            
                            if files and times:
                                plt.bar(range(len(files)), times, color='lightblue')
                                plt.xlabel('Image Files')
                                plt.ylabel('Extraction Time (s)')
                                plt.title('Message Extraction Time for Each Image')
                                plt.xticks(range(len(files)), files, rotation=45, ha='right')
                                plt.tight_layout()
                                
                                chart_path = os.path.join(CHARTS_DIR, f"report_decode_time_chart.png")
                                plt.savefig(chart_path)
                                plt.close()
                                print(f"📈 Generated extraction time chart: {os.path.basename(chart_path)}")
                else:
                    print("❌ No valid results files provided")
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter a number between 1 and 4.")