from PIL import Image
import numpy as np
import os 
import json 
import rasterio

PATH_TO_DATASET = '/home/alangomes/data/Itirapina/Itirapina/v2'

# tif_file = os.path.join(PATH_TO_DATASET, 'raw', 'images', '2011_263_13_1.jpg')

# # Read the image
# with rasterio.open(tif_file) as src:
#     image = src.read()

# img_shape = image.shape

# # Define the file path in your Google Drive
# output_file = os.path.join(PATH_TO_DATASET, 'data.txt')

# # Read the JSON file
# with open(output_file, 'r') as f:
#     data = json.load(f)

# PATH_TO_DATASET_V1 = '/home/alangomes/data/Itirapina/Itirapina/v1'

tif_file = os.path.join(PATH_TO_DATASET, 'raw', 'images', '1', '2011_263_13.jpg')
# tif_file = os.path.join(PATH_TO_DATASET_V1, 'raw', 'images', '2011_263_13_1.jpg')

# Read the image
with rasterio.open(tif_file) as src:
    image = src.read()

img_shape = image.shape

# Define the file path in your Google Drive
output_file = os.path.join(PATH_TO_DATASET, 'data.txt')

# Read the JSON file
with open(output_file, 'r') as f:
    data = json.load(f)

print("Data loaded successfully")

def extract_coord (pixel):
  numbers_str = pixel.strip('()').split(',')
  number1 = int(numbers_str[0])
  number2 = int(numbers_str[1])
  return number1, number2

def extract_class(data):
  return data['class_name'], data['class_id']

def create_mask_one_vs_all(specie, width, height, test_dict):
  mask = np.zeros((height, width), dtype=np.uint8)
  for pixel in data['data'].keys():
    i, j = extract_coord(pixel)
    class_name, class_id = extract_class(data['data'][pixel])
    file_name = data['data'][pixel]['file_name']
    is_test = file_name in test_dict
    # verificar pelo nome do arquivo
    # print(class_name, file_name)
    if class_name == specie and not is_test:
      mask[i, j] = 1
    elif class_name != specie and not is_test:
      mask[i, j] = 2
    elif class_name == specie and is_test:
      mask[i, j] = 3
    elif class_name != specie and is_test:
      mask[i, j] = 2
  return mask

def create_multi_class_mask(width, height, test_dict, n_class):
  mask = np.zeros((height, width), dtype=np.uint8)
  dict_class_mask = {}
  for pixel in data['data'].keys():
    i, j = extract_coord(pixel)
    class_name, class_id = extract_class(data['data'][pixel])
    file_name = data['data'][pixel]['file_name']
    is_test = file_name in test_dict

    if class_id not in dict_class_mask:
      dict_class_mask[class_id] = {
        "class_name": class_name,
        "train_id": class_id + 1,
        "test_id": class_id + 1 + n_class
      }

    class_offset = 0 if not is_test else n_class
    mask[i, j] = class_id + 1 + class_offset
  print(dict_class_mask)
  return mask

# aspidosperma_specie = 'aspidosperma' # A.tomentosum for v2
aspidosperma_specie = 'A.tomentosum' # A.tomentosum for v2
# rubiginosa_specie = 'm.rubiginosa' # M.rubiginosa for v2
rubiginosa_specie = 'M.rubiginosa'
dict_test_masks = {
  "mask_aspidosperma-2.bmp": 1,
  "mask_aspidosperma-3.bmp": 1,
  "mask_m.rubiginosa-2.bmp": 1,
  "mask_m.rubiginosa-3.bmp": 1,
  "mask_p.torta-2.bmp": 1,
  "mask_pouteria-2.bmp": 1
}

dict_test_masks_v2 = {
  "A.tomentosum_11.pgm": 1,
  "A.tomentosum_7.pgm": 1,
  "M.rubiginosa_34.pgm": 1,
  "C.brasiliensis_23.pgm": 1,
  "M.rubiginosa_4.pgm:": 1,
  "M.guianensis_21.pgm": 1,
  "P.torta_9.pgm": 1,
  "P.ramiflora_10.pgm": 1
}

def generate_and_save_mask(specie, test_dict):
  _, height, width = img_shape # need to run cell from beginning
  mask_np_array = create_mask_one_vs_all(specie, width, height, test_dict)
  mask_image = Image.fromarray(mask_np_array)
  mask_file_path = os.path.join(PATH_TO_DATASET, 'masks_convnet', f"{specie}_mask_int.png") # Example mask file
  mask_image.save(mask_file_path)

def generate_and_save_multi_class_mask(path, filename, test_dict):
  _, height, width = img_shape # need to run cell from beginning
  n_classes = len(data['classes'].keys())
  print('# Classes: ', n_classes)
  mask_np_array = create_multi_class_mask(width, height, test_dict, n_classes)
  mask_image = Image.fromarray(mask_np_array)
  mask_file_path = os.path.join(path, 'masks_convnet', f"{filename}.png") # Example mask file
  mask_image.save(mask_file_path)

def analyse_whole_mask(path, filename):
  # Choose an image file from the list (assuming it's a multi-band image)
  image_file = os.path.join(path, 'masks_convnet', f'{filename}.png')

  # Read the image
  with rasterio.open(image_file) as src:
      image = src.read()

  # Reshape the image data to have each pixel as a row (H*W, C)
  # Assuming the image has shape (C, H, W)
  reshaped_image = image.transpose(1, 2, 0).reshape(-1, image.shape[0])

  # Find the unique rows (unique pixel combinations)
  unique_pixels, counts = np.unique(reshaped_image, axis=0, return_counts=1)

  print(f"The unique RGB pixel types are:\n{unique_pixels}")
  print(f"There are {len(unique_pixels)} different unique RGB pixel types.")
  print(f"The counts for each unique pixel type are:\n{counts}")

generate_and_save_mask(aspidosperma_specie, dict_test_masks_v2)

generate_and_save_mask(rubiginosa_specie, dict_test_masks_v2)


# output_path = '/home/alangomes/data/Itirapina/Itirapina/v1'
# filename = "whole_mask_int_itirapina_v1"
# generate_and_save_multi_class_mask(output_path, filename)
# analyse_whole_mask(output_path, filename)

output_path = '/home/alangomes/data/Itirapina/Itirapina/v2'
filename = "whole_mask_int_itirapina_v2"
# filename = "A.tomentosum_mask_int"
generate_and_save_multi_class_mask(output_path, filename, dict_test_masks_v2)
analyse_whole_mask(output_path, filename)
