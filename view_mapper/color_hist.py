import cv2
import numpy as np
import os

def histogram_matching(source, reference):
    """Adjust the color of the source image to match the histogram of the reference image."""
    matched = np.zeros_like(source)
    for i in range(3):  # Assuming images are in RGB format
        source_hist, bins = np.histogram(source[:, :, i].ravel(), 256, [0, 256])
        reference_hist, _ = np.histogram(reference[:, :, i].ravel(), 256, [0, 256])

        # Cumulative distribution function (CDF)
        source_cdf = source_hist.cumsum()
        source_cdf = source_cdf / source_cdf[-1]  # Normalize
        
        reference_cdf = reference_hist.cumsum()
        reference_cdf = reference_cdf / reference_cdf[-1]  # Normalize

        # Create a mapping from source to reference
        mapping = np.interp(source_cdf, reference_cdf, np.arange(256))

        # Apply mapping to source channel
        matched[:, :, i] = cv2.LUT(source[:, :, i], mapping.astype(np.uint8))

    return matched

def process_folder(folder_path, reference_image_path, output_folder):
    """Adjust the color of all images in a folder to match a reference image."""
    # Load reference image
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"Error: Unable to load reference image from {reference_image_path}")
        return

    # Convert reference image to RGB
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)

        if not os.path.isfile(input_path):
            continue

        # Read the image
        source_image = cv2.imread(input_path)
        if source_image is None:
            print(f"Skipping {filename}: Unable to read image.")
            continue

        # Convert to RGB
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        # Match the histogram
        matched_image = histogram_matching(source_image, reference_image)

        # Convert back to BGR for saving
        matched_image = cv2.cvtColor(matched_image, cv2.COLOR_RGB2BGR)

        # Save the adjusted image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, matched_image)
        print(f"Processed and saved: {output_path}")

# Example usage
process_folder(
    folder_path='/home/nguyen/.code/statue_infusion/input_images', 
    reference_image_path='/home/nguyen/.code/statue_infusion/reference.jpg', 
    output_folder='/home/nguyen/.code/statue_infusion/output_images'
)
