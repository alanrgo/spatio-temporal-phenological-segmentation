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
