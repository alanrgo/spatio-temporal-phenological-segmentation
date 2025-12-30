import rasterio
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def create_pred_mask(mask_path, data, model, test_dataloader, device):
    # Use 'mask_train_test.png' to get the dimensions, as this is the image
    # from which the pixel coordinates for the dataset were derived.
    mask_train_test_file = os.path.join(mask_path, 'mask_train_test.png')

    with rasterio.open(mask_train_test_file) as src:
        image_height = src.height
        image_width = src.width

    print(f"Image Height: {image_height}")
    print(f"Image Width: {image_width}")

    # Define the class ID to color mapping (0-3 for the model's output classes)
    # using the colors defined in the 'data' dictionary for classes 0-3.
    class_id_to_color = {
        0: tuple(data["colors"]["rgbs"]["red"]),    # Model class 0 -> Red
        1: tuple(data["colors"]["rgbs"]["blue"]),   # Model class 1 -> Blue
        2: tuple(data["colors"]["rgbs"]["green"]),  # Model class 2 -> Green
        3: tuple(data["colors"]["rgbs"]["aqua"]),   # Model class 3 -> Aqua
    }

    print("Class ID to color mapping defined.")

    # Initialize a blank prediction mask image (black background)
    # 'device' is defined in cell r3Gn2lvmtiT1
    prediction_mask = torch.zeros((image_height, image_width, 3), dtype=torch.uint8, device=device)

    print("Blank prediction mask initialized.")

    # Get the original pixel coordinates for the test set in the same order
    # as they were added to `full_data_test` and `Y_test`.
    test_pixel_coords_ordered = []
    for pixel_position_key in data['data']['test'].keys():
        numbers_str = pixel_position_key.strip('()').split(',')
        i = int(numbers_str[0])
        j = int(numbers_str[1])
        test_pixel_coords_ordered.append((i, j))

    # Set the model to evaluation mode
    model.eval()

    # Collect all predictions from the test dataloader
    all_predictions = []
    with torch.no_grad():
        for x_batch, _ in test_dataloader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            predictions = outputs.argmax(1)  # Get predicted class IDs
            all_predictions.append(predictions.cpu()) # Move to CPU for concatenating

    all_predictions = torch.cat(all_predictions, dim=0).numpy()

    # Populate the prediction_mask based on the collected predictions and coordinates
    for idx, (i, j) in enumerate(test_pixel_coords_ordered):
        predicted_class = all_predictions[idx]
        color = class_id_to_color[predicted_class]
        prediction_mask[i, j, :] = torch.tensor(color, dtype=torch.uint8)

    print("Prediction mask populated with model predictions.")
    return prediction_mask, all_predictions

def overlay_mask_and_input(input_path, prediction_mask, output_path=None):
    # Choose a reference image to overlay the mask on
    # Using the first image from crop_filenames as an example
    any_file_name = "2016_03_30_crop.tif"
    reference_image_file = os.path.join(input_path, any_file_name)

    # Load the reference image
    with rasterio.open(reference_image_file) as src:
        original_image = src.read()

    # Normalize the original image data for display if necessary
    # Assuming it's a multi-band image, take the first three bands for RGB display
    img_display = original_image[:3, :, :].transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
    img_display = img_display / np.max(img_display) # Scale to [0, 1] for proper display

    # Move prediction_mask to CPU for matplotlib display and convert to numpy
    prediction_mask_np = prediction_mask.cpu().numpy()

    # Define the opacity for the mask (value between 0 and 1)
    mask_opacity = 0.9

    # Display the original image
    plt.figure(figsize=(10, 8))
    plt.imshow(img_display)

    # Overlay the prediction mask
    # Ensure the prediction_mask_np has the correct shape for overlay (H, W, 3)
    plt.imshow(prediction_mask_np, alpha=mask_opacity)

    plt.title(f'Prediction Mask Overlay on {os.path.basename(reference_image_file)}')
    plt.axis('off') # Hide axes for cleaner image display

    # Save overlay instead of showing
    output_path = os.path.join(output_path, "prediction_overlay.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return output_path

def creates_confusion_matrix(data, Y_test, all_predictions, output_path):
    # Ensure Y_test is a numpy array
    Y_test_np = Y_test.cpu().numpy()

    # Calculate the confusion matrix
    cm = confusion_matrix(Y_test_np, all_predictions)

    # Normalize the confusion matrix to get percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = list(data['classes'].values())

    # Create a heatmap of the confusion matrix
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes[0:4], yticklabels=classes[0:4])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Percentages)')
    # Save overlay instead of showing
    output_path = os.path.join(output_path, "confusion_matrix.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def generate_and_save_visualizations(data, Y_test, model, test_dataloader, device, input_path, output_path):
    pred_mask, all_predictions = create_pred_mask(input_path, data, model, test_dataloader, device)
    overlay_mask_and_input(input_path, pred_mask, output_path)
    creates_confusion_matrix(data, Y_test, all_predictions, output_path)

def plot_training_accs_along_epochs(train_accuracies, val_accuracies, output_path):
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy') # Changed test_accuracies to val_accuracies
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Val Accuracy')
    output_path = os.path.join(output_path, "train_and_val_accs.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def create_pred_mask_itirapina(
        data, model, 
        input_path,
        dict_test_mask, device, dataset='itirapinav2'
        ):
    # Assuming GOOGLE_DRIVE_PATH is defined and points to the image directory
    # Use one of the image files to get the dimensions
    if dataset == 'itirapinav2':
        sample_image_file = os.path.join(input_path, 'raw', 'images', '1', '2011_242_13.jpg')
    else:
        sample_image_file = os.path.join(input_path, 'raw', 'images', '2011_242_13_1.jpg')

    with rasterio.open(sample_image_file) as src:
        image_height = src.height
        image_width = src.width

    print(f"Image Height: {image_height}")
    print(f"Image Width: {image_width}")

    # Define the class ID to color mapping (0-5 to distinct RGB tuples)
    class_id_to_color = {
        0: (255, 0, 0),    # Red for Aspidosperma tomentosum
        1: (0, 255, 0),    # Green for Caryocar brasiliensis
        2: (0, 0, 255),    # Blue for Myrcia guinesis
        3: (255, 255, 0),  # Yellow for Miconia rubiginosa
        4: (0, 255, 255),  # Cyan for Pouteria ramiflora
        5: (255, 0, 255)   # Magenta for Pouteria Torta
    }

    print("Class ID to color mapping defined.")

    # Initialize a blank prediction mask image (black background)
    prediction_mask = torch.zeros((image_height, image_width, 3), dtype=torch.uint8, device=device)

    print("Blank prediction mask initialized.")
    
    for pixel_position in data['data']:
        numbers_str = pixel_position.strip('()').split(',')
        number1 = int(numbers_str[0])
        number2 = int(numbers_str[1])

        any_hour = '06'

        is_test_data = True if data['data'][pixel_position]['file_name'] in dict_test_mask else False

        if not is_test_data:
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
                    # Convert to integers and then to a float tensor
                    array_result = torch.tensor([int(num) for num in numbers_str], dtype=torch.float32)
                else:
                    # Corrected to a 1D float tensor of size 3
                    array_result = torch.zeros((3,), dtype=torch.float32)
                time_serie_pixels.append(array_result)
            time_serie_pixels = torch.stack(time_serie_pixels)
            pixels.append(time_serie_pixels)

        pixels = torch.stack(pixels)
        pixels = pixels.to(device) # Move the tensor to the GPU
        pred = model(pixels.unsqueeze(0))
        pred = torch.argmax(pred, dim=1)
        pred = pred.item()
        # Fix: Convert the color tuple to a torch.Tensor with the correct dtype before assignment
        prediction_mask[number1][number2] = torch.tensor(class_id_to_color[pred], dtype=torch.uint8, device=device)
    return prediction_mask

def overlay_mask_and_input_itirapina(input_path, prediction_mask, output_path, dataset = 'itirapinav2'):
    # Choose a reference image to overlay the mask on
    # Using the first image from crop_filenames as an example
    if dataset == 'itirapinav2':
        reference_image_file = os.path.join(input_path, 'raw', 'images', '1', '2011_242_13.jpg')
    else:
        reference_image_file = os.path.join(input_path, 'raw', 'images', '2011_242_13_1.jpg')

    # Load the reference image
    with rasterio.open(reference_image_file) as src:
        original_image = src.read()

    # Normalize the original image data for display if necessary
    # Assuming it's a multi-band image, take the first three bands for RGB display
    img_display = original_image[:3, :, :].transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
    img_display = img_display / np.max(img_display) # Scale to [0, 1] for proper display

    # Move prediction_mask to CPU for matplotlib display and convert to numpy
    prediction_mask_np = prediction_mask.cpu().numpy()

    # Define the opacity for the mask (value between 0 and 1)
    mask_opacity = 0.8

    # Display the original image
    plt.figure(figsize=(10, 8))
    plt.imshow(img_display)

    # Overlay the prediction mask
    # Ensure the prediction_mask_np has the correct shape for overlay (H, W, 3)
    plt.imshow(prediction_mask_np, alpha=mask_opacity)

    plt.title(f'Prediction Mask Overlay on {os.path.basename(reference_image_file)}')
    plt.axis('off') # Hide axes for cleaner image display
    output_path = os.path.join(output_path, "overlay_mask.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def creates_confusion_matrix_itirapina(model, data, dataloader_test, output_path, device):
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Normalize the confusion matrix to show percentages
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=data['classes'].values(), yticklabels=data['classes'].values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Percentages)')
    # Save overlay instead of showing
    output_path = os.path.join(output_path, "confusion_matrix.png")

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def generate_and_save_visualizations_itirapina(
        data, 
        dataset, 
        model, 
        test_dataloader, 
        device, 
        input_path, 
        output_path,
        train_accuracies,
        val_accuracies,
        dict_test_mask
        ):
    plot_training_accs_along_epochs(train_accuracies, val_accuracies, output_path)
    pred_mask = create_pred_mask_itirapina(data, model, 
        input_path,
        dict_test_mask, device, dataset)
    overlay_mask_and_input_itirapina(input_path, pred_mask, output_path, dataset)
    creates_confusion_matrix_itirapina(model, data, test_dataloader, output_path, device)
