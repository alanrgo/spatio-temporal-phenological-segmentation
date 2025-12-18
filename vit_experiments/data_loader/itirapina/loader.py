import torch
def load_itirapina_data(data):
    full_data = []
    Y = []
    full_data_test = []
    Y_test = []

    dict_class_to_index = {
        "A.tomentosum": 0,
        "C.brasiliensis": 1,
        "M.guianensis": 2,
        "M.rubiginosa": 3,
        "P.ramiflora": 4,
        "P.torta": 5
    }

    dict_test_masks = {
        "A.tomentosum_11.pgm": 1,
        "A.tomentosum_7.pgm": 1,
        "M.rubiginosa_34.pgm": 1,
        "M.rubiginosa_4.pgm:": 1,
        "P.torta_9.pgm": 1,
        "P.ramiflora_10.pgm": 1
    }

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

    for pixel_position in data['data']:
        numbers_str = pixel_position.strip('()').split(',')
        number1 = int(numbers_str[0])
        number2 = int(numbers_str[1])

        any_hour = '06'

        should_ignore = True if data['data'][pixel_position]['file_name'] in dict_mask_ignore else False
        is_test_data = True if data['data'][pixel_position]['file_name'] in dict_test_masks else False

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
                    array_result = torch.tensor([int(num) for num in numbers_str]) # Convert to integers and then to a NumPy array
                else:
                    array_result = torch.zeros((3,)) # Corrected to a 1D tensor of size 3
                time_serie_pixels.append(array_result)
            time_serie_pixels = torch.stack(time_serie_pixels)
            pixels.append(time_serie_pixels)

        pixels = torch.stack(pixels)
        if not is_test_data:
            full_data.append(pixels)
            Y.append(torch.tensor(dict_class_to_index[data['data'][pixel_position]['class_name']])) # Convert integer to tensor
        else:
            full_data_test.append(pixels)
            Y_test.append(torch.tensor(dict_class_to_index[data['data'][pixel_position]['class_name']])) # Convert integer to tensor


    full_data = torch.stack(full_data)
    Y = torch.stack(Y)
    full_data_test = torch.stack(full_data_test)
    Y_test = torch.stack(Y_test)
