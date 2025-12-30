import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_itirapina_data(data, dict_class_to_index, dict_test_masks):
    full_data = []
    Y = []
    full_data_test = []
    Y_test = []

    dict_mask_ignore = { # Comment the masks that will be used
    # "A.tomentosum_11.pgm": 1,
    # "A.tomentosum_7.pgm": 1,
    # "A.tomentosum_5.pgm": 1,
    # "C.brasiliensis_2.pgm": 1,
    # "C.brasiliensis_29.pgm": 1,
    # "C.brasiliensis_31.pgm": 1,
    # "C.brasiliensis_23.pgm": 1,
    # "M.guianensis_1.pgm": 1,
    # "M.guianensis_21.pgm": 1,
    # "C.brasiliensis_2.pgm": 1,
    # "C.brasiliensis_29.pgm": 1,
    # "C.brasiliensis_31.pgm": 1,
    # "C.brasiliensis_23.pgm": 1,
    # "M.rubiginosa_15.pgm": 1,
    # "M.rubiginosa_24.pgm": 1,
    # "M.rubiginosa_4.pgm:": 1,
    # "M.rubiginosa_32.pgm": 1,
    # "M.rubiginosa_34.pgm": 1,
    # "M.rubiginosa_28.pgm": 1,
    # "M.rubiginosa_6.pgm": 1,
    # "P.torta_18.pgm": 1,
    # "P.torta_17.pgm": 1,
    # "P.torta_3.pgm": 1,
    # "P.torta_9.pgm": 1,
    # "P.ramiflora_10.pgm": 1,
    # "P.ramiflora_8.pgm" 1,
    }

    debug_masks = {}
    for pixel_position in data['data']:
        numbers_str = pixel_position.strip('()').split(',')
        number1 = int(numbers_str[0])
        number2 = int(numbers_str[1])

        any_hour = '06'

        should_ignore = True if data['data'][pixel_position]['file_name'] in dict_mask_ignore else False
        should_ignore = should_ignore or data['data'][pixel_position]['class_name'] == 'no_class'
        is_test_data = True if data['data'][pixel_position]['file_name'] in dict_test_masks else False
        
        file_name = data['data'][pixel_position]['file_name']
        if file_name not in debug_masks:
            print(file_name, "is test: ", is_test_data)
            debug_masks[file_name] = 0
        else:
            debug_masks[file_name] += 1
        if should_ignore:
            continue
        time_serie_size = len(data['data'][pixel_position]['hours'][any_hour])

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
            default_hour = 12
            for pixel in region:
                pixel_key = f'({pixel[0]},{pixel[1]})'
                if pixel_key in data['data']:
                    hour_key = default_hour
                    hour_key = str(hour_key).zfill(2)
                    rgb_str = data['data'][pixel_key]['hours'][hour_key][time_serie_index]
                    numbers_str = rgb_str.strip('[]').split(',') # Remove brackets and split by comma
                    array_result = torch.tensor([int(num) for num in numbers_str], dtype=torch.float32) # Convert to float tensor
                else:
                    array_result = torch.zeros((3,), dtype=torch.float32) # Corrected to a 1D tensor of size 3
                time_serie_pixels.append(array_result)
            time_serie_pixels = torch.stack(time_serie_pixels)
            pixels.append(time_serie_pixels)

        pixels = torch.stack(pixels)
        if not is_test_data:
            full_data.append(pixels)
            Y.append(torch.tensor(dict_class_to_index[data['data'][pixel_position]['class_name']], dtype=torch.long)) # Convert integer to long tensor for labels
        else:
            full_data_test.append(pixels)
            Y_test.append(torch.tensor(dict_class_to_index[data['data'][pixel_position]['class_name']], dtype=torch.long)) # Convert integer to long tensor for labels


    full_data = torch.stack(full_data)
    Y = torch.stack(Y)
    full_data_test = torch.stack(full_data_test)
    Y_test = torch.stack(Y_test)

    # Converte para numpy (necessário para usar train_test_split)
    X_np = full_data.numpy()
    y_np = Y.numpy()
    # Divide o conjunto de treino em treino (80%) e validação (20%)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_np, y_np, test_size=0.2, stratify=y_np, random_state=42
    )

    # Converte de volta para tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_val = torch.tensor(Y_val, dtype=torch.long)
    return X_train, X_val, Y_train, Y_val, full_data_test, Y_test
