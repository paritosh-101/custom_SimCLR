from PIL import Image
import numpy as np

def compare_channels(image_path):
    # Load the image
    img = Image.open(image_path)
    channels = img.split()

    # Convert channels to numpy arrays for comparison
    arrays = [np.array(channel) for channel in channels]

    # Compare channels
    diff_01 = np.mean(np.abs(arrays[0] - arrays[1]))
    diff_02 = np.mean(np.abs(arrays[0] - arrays[2]))
    diff_12 = np.mean(np.abs(arrays[1] - arrays[2]))

    print(f"Average difference between channel 0 and 1: {diff_01}")
    print(f"Average difference between channel 0 and 2: {diff_02}")
    print(f"Average difference between channel 1 and 2: {diff_12}")

    # Suggest grayscale conversion if differences are small
    if diff_01 < threshold and diff_02 < threshold and diff_12 < threshold:
        print("Channels are quite similar. Consider converting to grayscale.")
    else:
        print("Channels contain different information.")

# Set your image path and threshold
image_path = 'Data_SIMCLR_balanced/DD/3DIMG_01DEC2020_0000_L1C_SGP_V01R00_B3_event09_2020_DD.png'  # Replace with your image path
threshold = 10  # Set a threshold for difference

compare_channels(image_path)