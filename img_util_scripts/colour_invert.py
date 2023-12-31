from PIL import Image
import os

main_folder = 'Data_SIMCLR_balanced'

def invert_colors(image_path, save_path):
    with Image.open(image_path) as img:
        inverted_img = Image.eval(img, lambda x: 255 - x)
        inverted_img.save(save_path)

for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subfolder_path, file)
                # Uncomment the next line if you want to overwrite the original image
                invert_colors(file_path, file_path)
                # Or use this line to save the inverted image as a new file
                # invert_colors(file_path, os.path.join(subfolder_path, f"inverted_{file}"))