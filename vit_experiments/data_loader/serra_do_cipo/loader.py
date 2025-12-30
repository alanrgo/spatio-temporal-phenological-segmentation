import os 
import torch
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_raw_data(path = None, data_custom = None):
  PATH_TO_READ_FILE = path
  data_custom = 'data.txt' if data_custom is None else data_custom

  # Define the file path in your Google Drive
  output_file = os.path.join(PATH_TO_READ_FILE, data_custom)

  # Read the JSON file
  with open(output_file, 'r') as f:
      data = json.load(f)

  print("Data loaded successfully from: ", output_file)
  return data

def load_sc_data(data):
  full_data = []
  Y = []

  for pixel_position in data['data']['train']:
    if data['data']['train'][pixel_position]['class'] == -1:
      continue
    numbers_str = pixel_position.strip('()').split(',')
    number1 = int(numbers_str[0])
    number2 = int(numbers_str[1])

    time_serie_size = len(data['data']['train'][pixel_position]['timeserie'])
    # print(data['data']['train'][pixel_position])
    # Upper pixels
    upper_left = (number1 - 1, number2 - 1)
    upper = (number1 - 1, number2)
    upper_right = (number1 - 1, number2 + 1)

    # Same row
    left = (number1, number2 - 1)
    right = (number1, number2 + 1)

    # Lower pixels
    lower_left = (number1 + 1, number2 - 1)
    lower_right = (number1 + 1, number2 + 1)
    lower = (number1 + 1, number2)


    region = [
      upper_left,
      upper,
      upper_right,
      left,
      (number1, number2), # main pixel centered
      right,
      lower_left,
      lower,
      lower_right
    ]

    pixels = []
    # print('time_serie_size', time_serie_size)
    for time_serie_index in range(time_serie_size):
      time_serie_pixels = []
      for pixel in region:
        pixel_key = f'({pixel[0]},{pixel[1]})'
        if pixel_key in data['data']['train']:
          # print(data['data']['train'][pixel_key])
          rgb_str = data['data']['train'][pixel_key]['rgbs'][time_serie_index]
          numbers_str = rgb_str.strip('[]').split() # Remove brackets and split by spaces
          array_result = torch.tensor([int(num) for num in numbers_str], dtype=torch.float32) # Convert to float tensor
        else:
          array_result = torch.zeros((3,), dtype=torch.float32) # Corrected to a 1D tensor of size 3
        time_serie_pixels.append(array_result)
      time_serie_pixels = torch.stack(time_serie_pixels)
      pixels.append(time_serie_pixels)

    pixels = torch.stack(pixels)
    full_data.append(pixels)
    Y.append(torch.tensor(int(data['data']['train'][pixel_position]['class']), dtype=torch.long)) # Labels as long tensor

  full_data = torch.stack(full_data)
  Y = torch.stack(Y)
  print(full_data.shape)
  print(Y.shape)

  full_data_test = []
  Y_test = []
  class_map = {
      "4": 1,
      "5": 2,
      "6": 0,
      "7": 3,
  }

  for pixel_position in data['data']['test']:
    if data['data']['test'][pixel_position]['class'] == -1:
      continue
    numbers_str = pixel_position.strip('()').split(',')
    number1 = int(numbers_str[0])
    number2 = int(numbers_str[1])

    time_serie_size = len(data['data']['test'][pixel_position]['timeserie'])

      # Upper pixels
    upper_left = (number1 - 1, number2 - 1)
    upper = (number1 - 1, number2)
    upper_right = (number1 - 1, number2 + 1)

    # Same row
    left = (number1, number2 - 1)
    right = (number1, number2 + 1)

    # Lower pixels
    lower_left = (number1 + 1, number2 - 1)
    lower_right = (number1 + 1, number2 + 1)
    lower = (number1 + 1, number2)


    region = [
      upper_left,
      upper,
      upper_right,
      left,
      (number1, number2), # main pixel centered
      right,
      lower_left,
      lower,
      lower_right
    ]

    pixels = []
    for time_serie_index in range(time_serie_size):
      time_serie_pixels = []
      for pixel in region:
        pixel_key = f'({pixel[0]},{pixel[1]})'
        if pixel_key in data['data']['test']:
          rgb_str = data['data']['test'][pixel_key]['rgbs'][time_serie_index]
          numbers_str = rgb_str.strip('[]').split() # Remove brackets and split by spaces
          array_result = torch.tensor([int(num) for num in numbers_str], dtype=torch.float32) # Convert to float tensor
        else:
          array_result = torch.zeros((3,), dtype=torch.float32) # Corrected to a 1D tensor of size 3
        time_serie_pixels.append(array_result)
      time_serie_pixels = torch.stack(time_serie_pixels)
      pixels.append(time_serie_pixels)

    pixels = torch.stack(pixels)
    full_data_test.append(pixels)
    origin_class = str(data['data']['test'][pixel_position]['class'])
    Y_test.append(torch.tensor(class_map[origin_class], dtype=torch.long)) # Labels as long tensor


  full_data_test = torch.stack(full_data_test)
  Y_test = torch.stack(Y_test)
  print(full_data_test.shape)
  print(Y_test.shape)

  # Converte para numpy (necessário para usar train_test_split)
  X_np = full_data.numpy()
  y_np = Y.numpy()

  # Divide o conjunto de treino em treino (80%) e validação (20%)
  X_train, X_val, Y_train, Y_val = train_test_split(
      X_np, y_np, test_size=0.2, stratify=y_np, random_state=42
  )

  # Converte de volta para tensores com dtypes corretos
  X_train = torch.tensor(X_train, dtype=torch.float32)
  X_val = torch.tensor(X_val, dtype=torch.float32)
  Y_train = torch.tensor(Y_train, dtype=torch.long)
  Y_val = torch.tensor(Y_val, dtype=torch.long)

  return X_train, Y_train, X_val, Y_val, full_data_test, Y_test

def create_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test):
  # Cria datasets
  train_dataset = TensorDataset(X_train, Y_train)
  val_dataset = TensorDataset(X_val, Y_val)
  dataset_test = TensorDataset(X_test, Y_test)

  # Cria DataLoaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
  test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

  return train_loader, val_loader, test_loader
