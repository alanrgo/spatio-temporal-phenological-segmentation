import os

import rasterio
import os
import json
import re
import time

PATH_TO_RAW_ITIRAPINA_V2_IMAGES = '/home/alangomes/data/Itirapina/Itirapina/v2/raw/images/1'

def populate_data_coord(data, mask_files):
    from PIL import Image
    import numpy as np
    import os
    import torch

    # mask_file and classes are expected to be defined in previous cells.
    # mask_file is defined in fBMPfp-UITeA as: os.path.join(GOOGLE_DRIVE_PATH, '..', '..', 'masks')

    non_black_pixel_counts = {}
    data['info'] = {}

    for class_name in classes:
        # List files in the actual mask directory (mask_file from fBMPfp-UITeA)
        all_mask_files_in_dir = os.listdir(mask_files)

        # Filter files for the current class
        # Assuming class names directly match part of the filename, e.g., 'A.tomentosum' in 'A.tomentosum_11.pgm'
        class_mask_files = [f for f in all_mask_files_in_dir if class_name in f]

        if not class_mask_files:
            print(f"No mask files found for class: {class_name}")
            continue

        print(f"Processing class: {class_name} with {len(class_mask_files)} mask files.")
        file_counter = 0

        if class_name not in data['info']:
            data['info'][class_name] = {}
        data['info'][class_name]['num_instances'] = len(class_mask_files)
        class_non_black_count = 0
        for file_name in class_mask_files:
            full_mask_path = os.path.join(mask_files, file_name)
            try:
                mask_pil = Image.open(full_mask_path)
                if mask_pil.mode != 'L':
                    mask_pil = mask_pil.convert('L') # Ensure it's grayscale
                mask_array = np.array(mask_pil)

                # Convert numpy array to torch tensor
                mask_tensor = torch.from_numpy(mask_array)

                # 2️⃣ Get coordinates of non-black pixels
                coords = torch.nonzero(mask_tensor, as_tuple=False)  # shape: (N, 2)

                class_non_black_count += len(coords)

                if 'instances' not in data["info"][class_name]:
                    data["info"][class_name]['instances'] = {}

                if file_counter not in data["info"][class_name]['instances']:
                    data["info"][class_name]['instances'][file_counter] = {}

                data["info"][class_name]['instances'][file_counter]['num_pixels'] = len(coords)

                for i, j in coords.tolist():  # small loop, only over non-black pixels
                    key = f'({i},{j})'
                    data['data'][key] = {
                        'class_name': class_name,
                        'class_id': classes.index(class_name),
                        'file_name': file_name,
                        'hours': {}
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
                        if neighbor_key not in data['data']:
                            data['data'][neighbor_key] = {
                                'class_name': 'no_class',
                                'class_id': -1,
                                'file_name': file_name,
                                'hours': {}
                            }
            
            except Exception as e:
                print(f"Error processing mask file {full_mask_path}: {e}")
            file_counter += 1
        data['info'][class_name]['num_pixels'] = class_non_black_count
    return data

  # This line will now cause a problem if it's uncommented, as it's trying to assign
  # a count that's not being accumulated for this specific cell's purpose.
  # I'm commenting it out as the main goal of this cell was to show `coords`.
  # non_black_pixel_counts[class_name] = class_non_black_count

data = {
    "name": "itirapina v2",
    "classes": {
        "0": "Aspidosperma tomentosum",
        "1": "Caryocar brasiliensis",
        "2": "Myrcia guinesis",
        "3": "Miconia rubiginosa",
        "4": "Pouteria ramiflora",
        "5": "Pouteria Torta",
    },
    "data": {

    }
}


mask_files = os.path.join(PATH_TO_RAW_ITIRAPINA_V2_IMAGES, '..', '..', 'masks') # Example mask file
classes = ['A.tomentosum', 'C.brasiliensis', 'M.guianensis', 'C.brasiliensis', 'M.rubiginosa', 'P.torta', 'P.ramiflora']

data = populate_data_coord(data, mask_files)

crop_filenames = sorted(os.listdir(PATH_TO_RAW_ITIRAPINA_V2_IMAGES))
print(crop_filenames)

for filename in crop_filenames:

  # Choose an image file from the list (assuming it's a multi-band image)
  image_file = os.path.join(PATH_TO_RAW_ITIRAPINA_V2_IMAGES, filename)

  hour = filename.split('_')[-1].split('.')[0]

  # Read the image
  with rasterio.open(image_file) as src:
      image = src.read()

  # Reshape the image data to have each pixel as a row (H*W, C)
  # Assuming the image has shape (C, H, W)
  reshaped_image = image.transpose(1, 2, 0)

  # # 3️⃣ Iterate efficiently only over relevant pixels
  for coord_str in data['data'].keys():
    match = re.match(r"\(([^,]+),([^)]+)\)", coord_str)

    if match:
      i, j = match.groups()
    else:
      print("No match found")

    pixel = reshaped_image[int(i), int(j)].tolist()
    if hour not in data['data'][coord_str]['hours']:
      data['data'][coord_str]['hours'][hour] = []
    data['data'][coord_str]['hours'][hour].append(str(pixel))

# Define the file path in your Google Drive
output_file = os.path.join(PATH_TO_RAW_ITIRAPINA_V2_IMAGES, '..', '..', 'data_9.txt')

# Save the dictionary as a JSON file
with open(output_file, 'w') as f:
    json.dump(data, f)