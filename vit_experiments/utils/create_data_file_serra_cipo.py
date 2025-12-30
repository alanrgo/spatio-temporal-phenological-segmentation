import os

import rasterio
import os
import json
import re
import time

PATH_TO_RAW_SERRA_CIPO = '/home/alangomes/data/Dados_serra_cipo/Dados_serra_cipó/Imagens_Cedro_ePhenoWS'

print(os.listdir(PATH_TO_RAW_SERRA_CIPO))

def extract_date_from_filename(filename):
  """Extracts the date string (YYYY_MM_DD) from a filename."""
  parts = filename.split('.')
  if parts:
    name_without_extension = parts[0]
    date_parts = name_without_extension.split('_')
    if len(date_parts) >= 3:
      return '_'.join(date_parts[:3])
  return None

def filter_filenames(filenames):
  """Filters filenames to include 'crop' and exclude 'Cópia de'."""
  filtered_list = [f for f in filenames if 'crop' in f and 'Cópia de' not in f]
  return filtered_list


def has_c(filename):
  return 'c' in filename

def has_mask(filename):
  return 'mask' in filename

file_list = os.listdir(PATH_TO_RAW_SERRA_CIPO)
crop_filenames = filter_filenames(file_list)
print(crop_filenames)

dates = [extract_date_from_filename(filename) for filename in crop_filenames]

data = {
    "name": "serra do cipo",
    "timeserie_size": len(crop_filenames),
    "timeserie": dates,
    "classes": {
        "0": "Bowdichia virgilioides",
        "1": "Eremanthus erythropappus",
        "2": "Collection of evergreen species",
        "3": "Vochysia cinnamomea",
        "4": "Eremanthus erythropappus",
        "5": "Collection of evergreen species",
        "6": "Bowdichia virgilioides",
        "7": "Vochysia cinnamomea"
    },
    "colors": {
        "description": {
          "red": 0,
          "blue": 1,
          "green": 2,
          "aqua": 3,
          "purple": 4,
          "yellow": 5,
          "orange": 6,
          "white": 7
        },
        "rgbs": {
          "red": [255, 0, 0],
          "blue": [0, 0, 255],
          "green": [0, 255, 0],
          "aqua": [0, 255, 255],
          "purple": [128, 0, 128],
          "yellow": [255, 255, 0],
          "orange": [255, 153, 51],
          "white": [255, 255, 255]
        },
        "rgb_to_class": {
          "[255, 0, 0]": 0,
          "[0, 0, 255]": 1,
          "[0, 255, 0]": 2,
          "[0, 255, 255]": 3,
          "[128, 0, 128]": 4,
          "[255, 255, 0]": 5,
          "[255, 153, 51]": 6,
          "[255, 255, 255]": 7
        },
        "split": {
          "train": [0, 1, 2, 3],
          "test": [4, 5, 6, 7]
        }
    },
    "data": {
        "train": {},
        "test": {}
    }
}

# Choose an image file from the list (assuming it's a multi-band image)
image_file = os.path.join(PATH_TO_RAW_SERRA_CIPO, 'mask_train_test.png')

# Read the image
with rasterio.open(image_file) as src:
    image = src.read()

# Reshape the image data to have each pixel as a row (H*W, C)
# Assuming the image has shape (C, H, W)
reshaped_image = image.transpose(1, 2, 0)

def is_train(pixel):
  if str(pixel) in data["colors"]["rgb_to_class"]:
    pixel_class = data["colors"]["rgb_to_class"][str(pixel)]
    return pixel_class in data["colors"]["split"]["train"]
  else:
    return False

def is_test(pixel):
  if str(pixel) in data["colors"]["rgb_to_class"]:
    pixel_class = data["colors"]["rgb_to_class"][str(pixel)]
    return pixel_class in data["colors"]["split"]["test"]
  else:
    return False
  
import torch

# Assume reshaped_image is (H, W, 3)
if not isinstance(reshaped_image, torch.Tensor):
    reshaped_image = torch.tensor(reshaped_image)

H, W, _ = reshaped_image.shape

# 1️⃣ Create a mask for non-black pixels
mask = torch.any(reshaped_image != 0, dim=-1)  # (H, W) boolean

# 2️⃣ Get coordinates of non-black pixels
coords = torch.nonzero(mask, as_tuple=False)  # shape: (N, 2)
# coords[n] = [i, j]

# 3️⃣ Iterate efficiently only over relevant pixels
for i, j in coords.tolist():  # small loop, only over non-black pixels
    pixel = reshaped_image[i, j].tolist()  # e.g. [R, G, B]
    category = 'train' if is_train(pixel) else 'test'
    key = f'({i},{j})'
    pixel_key = str(pixel)
    data_class = data["colors"]["rgb_to_class"][pixel_key]

    data["data"][category][key] = {
        "class": data_class,
        "rgbs": [],
        "timeserie": []
    }
    # Upper pixels
    upper_left = (i - 1, j - 1)
    upper = (i - 1, j)
    upper_right = (i - 1, j + 1)

    # Same row
    left = (i, j - 1)
    right = (i, j + 1)

    # Lower pixels
    lower_left = (i + 1, j - 1)
    lower_right = (i + 1, j + 1)
    lower = (i + 1, j)
    region = [
        upper_left,
        upper,
        upper_right,
        left,
        right,
        lower_left,
        lower,
        lower_right
    ]
    for neighbor in region:
        number1 = neighbor[0]
        number2 = neighbor[1]
        neighbor_key = f'({number1},{number2})'
        if neighbor_key not in data['data'][category]:
            data['data'][category][neighbor_key] = {
                "class": -1,
                "rgbs": [],
                "timeserie": []
            }
    

for filename in crop_filenames:

  # Choose an image file from the list (assuming it's a multi-band image)
  image_file = os.path.join(PATH_TO_RAW_SERRA_CIPO, filename)

  # Read the image
  with rasterio.open(image_file) as src:
      image = src.read()

  # Reshape the image data to have each pixel as a row (H*W, C)
  # Assuming the image has shape (C, H, W)
  first_image_reshaped = image.transpose(1, 2, 0)

  # 3️⃣ Iterate efficiently only over relevant pixels
  for i, j in coords.tolist():  # small loop, only over non-black pixels
    pixel = reshaped_image[i, j].tolist()  # e.g. [R, G, B]
    category = "train" if is_train(pixel) else "test"

    key = f'({i},{j})'
    pixel_key = str(pixel)
    time_serie_pixel = str(first_image_reshaped[i][j])
    data_class = data["colors"]["rgb_to_class"][pixel_key]

    timeserie = extract_date_from_filename(filename)
    already_saved = timeserie in data["data"][category][key]['timeserie']
    if not already_saved:
        data["data"][category][key]['rgbs'].append(time_serie_pixel)
        data["data"][category][key]['timeserie'].append(extract_date_from_filename(filename))

    upper_left = (i - 1, j - 1)
    upper = (i - 1, j)
    upper_right = (i - 1, j + 1)

    # Same row
    left = (i, j - 1)
    right = (i, j + 1)

    # Lower pixels
    lower_left = (i + 1, j - 1)
    lower_right = (i + 1, j + 1)
    lower = (i + 1, j)
    region = [
        upper_left,
        upper,
        upper_right,
        left,
        right,
        lower_left,
        lower,
        lower_right
    ]
    for neighbor in region:
        number1 = neighbor[0]
        number2 = neighbor[1]
        neighbor_key = f'({number1},{number2})'
        time_serie_pixel_neighbor = str(first_image_reshaped[number1][number2])
        timeserie = extract_date_from_filename(filename)

        if timeserie in data["data"][category][neighbor_key]['timeserie']:
           continue
        data["data"][category][neighbor_key]['rgbs'].append(time_serie_pixel_neighbor)
        data["data"][category][neighbor_key]['timeserie'].append(extract_date_from_filename(filename))

# Define the file path in your Google Drive
output_file = os.path.join(PATH_TO_RAW_SERRA_CIPO, '..', 'data_9.txt')

# Save the dictionary as a JSON file
with open(output_file, 'w') as f:
    json.dump(data, f)