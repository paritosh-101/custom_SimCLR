from PIL import Image
import numpy as np

image_path = 'D:\_workspace\ISRO_SimCLR\SimCLR\Data_SIMCLR_balanced\DD\\3DIMG_01DEC2020_0130_L1C_SGP_V01R00_B3_event09_2020_DD.png'  # Replace with the path to your image
img = Image.open(image_path)

# Image resolution (dimensions)
width, height = img.size
print(f"Resolution: {width}x{height} pixels")

# Number of channels
# 'L' for grayscale images, 'RGB' for true color images, 'RGBA' for true color with alpha channel, etc.
mode = img.mode
num_channels = len(mode)
print(f"Color mode: {mode}, Number of channels: {num_channels}")

# Image format (like JPEG, PNG, etc.)
format = img.format
print(f"Format: {format}")