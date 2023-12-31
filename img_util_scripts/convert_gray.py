from PIL import Image
import os

main_folder = 'Data_SIMCLR_balanced'  # Replace with your main folder path

def convert_to_grayscale(image_path, save_path):
    with Image.open(image_path) as img:
        grayscale_img = img.convert('L')  # Convert to grayscale
        grayscale_img.save(save_path)

for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subfolder_path, file)
                # Uncomment the next line if you want to overwrite the original image
                convert_to_grayscale(file_path, file_path)
                # Or use this line to save the grayscale image as a new file
                # convert_to_grayscale(file_path, os.path.join(subfolder_path, f"grayscale_{file}"))