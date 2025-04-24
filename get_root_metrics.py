import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.morphology import skeletonize
import os
import csv
import re
import time

# Original Image Dimensions: 2550x2273 pixels

# Takes ~4.5 seconds on average to process a single image

# TODO: Auto write all the processed data so far to a CSV file (if we want to halt processing or an error occurs midway)

def process_root_mask(mask):
    # Calculate root area, length, diameter, and volume
    area = np.sum(mask)
    skeleton = skeletonize(mask)
    length = np.sum(skeleton)
    diameter = 2 * area / length if length > 0 else 0
    volume = np.pi * (diameter / 2) ** 2 * length
    return area, length, diameter, volume

def resize_image(image, target_size=(1600, 1600)): 
    # Resize image to 1600x1600 pixels while maintaining aspect ratio
    h, w = image.shape[:2]
    if h == target_size[0] and w == target_size[1]:
        return image
    
    # Calculate scaling factor
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create black canvas
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Calculate position to paste resized image
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    
    # Paste resized image onto black canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def analyze_roots(image_path, model, conf_threshold=0.3):
    # Analyze roots in the image using the YOLO model
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Resize image if it's not 1600x1600px
    image = resize_image(image)
    
    results = model(image, conf=conf_threshold)
    
    if results[0].masks is None:
        print(f"No roots detected in {image_path} with confidence threshold {conf_threshold}")
        return {
            "Root Count": "0",
            "Total Root Area": "0",
            "Total Root Length": "0",
            "Average Root Diameter": "0",
            "Average Root Length": "0",
            "Total Root Volume": "0"
        }, image
    
    root_count = len(results[0].masks)
    total_area = 0
    total_length = 0
    diameters = []
    total_volume = 0
    
    masked_image = image.copy()
    
    for idx, (mask, box, conf) in enumerate(zip(results[0].masks, results[0].boxes.xyxy, results[0].boxes.conf)):
        mask_array = mask.data.cpu().numpy().squeeze()
        area, length, diameter, volume = process_root_mask(mask_array)
        total_area += area
        total_length += length
        diameters.append(diameter)
        total_volume += volume
        
        masked_image[mask_array > 0] = [0, 255, 0]
        
        # x, y = int(box[0]), int(box[1]) # Remove the confidence text on masked image
        # cv2.putText(masked_image, f"{conf:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    average_diameter = np.mean(diameters) if diameters else 0
    average_length = total_length / root_count if root_count > 0 else 0

    return {
        "Root Count": f"{root_count}",
        "Total Root Area": f"{total_area:.2f}",
        "Total Root Length": f"{total_length:.2f}",
        "Average Root Diameter": f"{average_diameter:.2f}",
        "Average Root Length": f"{average_length:.2f}",
        "Total Root Volume": f"{total_volume:.2f}"
    }, masked_image

def extract_tube_and_depth(filename):
    # Convert filename to lowercase for case-insensitive matching
    # Extract tube ID and depth from the filename
    filename_lower = filename.lower()
    match = re.search(r'_t(\d+)_l(\d+)', filename_lower)
    if match:
        tube_id = match.group(1)
        depth = int(match.group(2))
        return tube_id, depth
    return None, None

def write_csv_data(csv_data, csv_file_path='root_area_results.csv'):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'tube_id'] + [f'depth_{i}' for i in range(1, 8)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tube_data in csv_data.values():
            writer.writerow(tube_data)

def write_skipped_images(skipped_images, skipped_csv_file_path='skipped_images_log.csv'):
    with open(skipped_csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped_images)

def main():
    start_time = time.time()
    model_path = "./train41_best.pt"
    test_data_dir = "./test_data"
    output_dir = "./masks"
    conf_threshold = 0.3
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path).to(device)
    
    image_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Prepare data for CSV
    csv_data = {}
    skipped_images = []  # List to track skipped/unreadable images
    
    for image_file in image_files:
        image_path = os.path.join(test_data_dir, image_file)
        output_image_path = os.path.join(output_dir, f"masked_{image_file}")
        
        try:
            root_metrics, masked_image = analyze_roots(image_path, model, conf_threshold)
            
            try:
                cv2.imwrite(output_image_path, masked_image)
                print(f"Masked image saved to {output_image_path}")
            except Exception as e:
                print(f"Failed to save masked image for {image_file}: {e}")
            
            print(f"\nRoot Metrics for {image_file}:")
            print("-" * 50)
            for key, value in root_metrics.items():
                print(f"{key:<20}: {value} {'pixels²' if 'Area' in key else 'pixels' if 'Length' in key or 'Diameter' in key else 'pixels³' if 'Volume' in key else ''}")
            print("-" * 50)
            
            # Extract tube ID and depth
            tube_id, depth = extract_tube_and_depth(image_file)
            if tube_id and depth:
                if tube_id not in csv_data:
                    csv_data[tube_id] = {'image_name': image_file, 'tube_id': tube_id}
                csv_data[tube_id][f'depth_{depth}'] = root_metrics['Total Root Area']
                
                # Write CSV data after each successful image processing
                write_csv_data(csv_data)
        
        except ValueError as e:
            # Log unreadable/skipped images
            print(f"Skipping unreadable image {image_file}: {e}")
            skipped_images.append({'image_name': image_file, 'error': str(e)})
            # Write skipped images log after each skipped image
            write_skipped_images(skipped_images)
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print(f"CSV file saved to root_area_results.csv")
    print(f"Skipped images log saved to skipped_images_log.csv")
    print(f"Device used for inference: {model.device}")
    print(f"Confidence threshold used for predictions: {conf_threshold}")
    print(f"Total script processing time: {total_time:.2f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    main()