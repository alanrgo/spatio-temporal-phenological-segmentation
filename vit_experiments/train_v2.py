from data_loader.serra_do_cipo.loader import load_raw_data
import os
import argparse

from utils.vit_setup_loader import load_config
from data_loader.itirapina.loader import load_itirapina_data
from utils.create_loaders import create_loaders
from models.vit import VisionTransformer
import torch
import torch.nn as nn
import torch.optim as optim

from utils.create_output_folder import create_folder_if_needed
from utils.visualize_serra_cipo import generate_and_save_visualizations_itirapina
from utils.evaluate import evaluate
from utils.metrics import save_metrics

parser = argparse.ArgumentParser(description='Vit Experiments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str,
                    required=True,
                    help='serra_do_cipo, itirapinav1, ou itirapinav2')
parser.add_argument('--input_path', type=str,
                    required=True,
                    help='path/to/input/files')
parser.add_argument('--data_file', type=str, required=False, help='name_of_file')
parser.add_argument('--setup_path', type=str,
                    required=True,
                    help='path/to/setups')
parser.add_argument('--output', type=str,
                    required=True,
                    help='path/to/outputs')
args = parser.parse_args()

dataset = args.dataset
input_path = args.input_path
setup_path = args.setup_path
output_path = args.output
data_file = args.data_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read data
data = load_raw_data(input_path, data_file)
# Ler dicionario de classes de itirapina

path = os.path.join(setup_path, 'setup_common_training.yaml')
loaded_config = load_config(path)
print(loaded_config)

classes = loaded_config['classes']
masks = loaded_config['masks']
dict_classes = {species: idx for idx, species in enumerate(classes)}
masks_dict = {mask: 1 for _, mask in enumerate(masks)}

X_train, X_val, Y_train, Y_val, X_test, Y_test = load_itirapina_data(data, dict_classes, masks_dict)
train_loader, val_loader, test_loader  = create_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test)

print("===== Shapes of training sets ====")
print("X_train: ", X_train.shape)
print("X_val: ", X_val.shape)
print("Y_train: ", Y_train.shape)
print("Y_val: ", Y_val.shape)
print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)
print("===== Shapes of training sets ==== \n")

setup_path = setup_path
setup_list = sorted(os.listdir(setup_path))
print(setup_list)

dict_ignore = {
    'setup_common_training.yaml': 1
}

for setup in setup_list:
    PATH_OUTPUTFILE = output_path
    
    if setup in dict_ignore:
        continue
    setup_file_path = os.path.join(setup_path, setup)
    loaded_config = load_config(setup_file_path)
    print(loaded_config)

    EXPERIMENT_NAME = loaded_config['experiment_name']
    folder_path = os.path.join(PATH_OUTPUTFILE, EXPERIMENT_NAME)
    if os.path.exists(folder_path):
        continue

    REGION_SIZE = loaded_config['custom_setup']['region']
    SEQUENCE_ORDER = loaded_config['custom_setup']['sequence_order']
    FEATURE_ARRANGEMENT = loaded_config['custom_setup']['arrangement']
    NORMALIZED_RGB = loaded_config['custom_setup']['normalized_rgb']
    CHANNELS = loaded_config['model']['channels']

    if 'pos_encoding' in loaded_config['custom_setup']:
        POS_ENCODING_ENABLED = loaded_config['custom_setup']['pos_encoding']
    else:
        POS_ENCODING_ENABLED = True

    if 'aggregation' in loaded_config['custom_setup']:
        POOL_TYPE = loaded_config['custom_setup']['aggregation']
    else:
        POOL_TYPE = 'cls'

    SEQ_LEN = loaded_config['training']['seq_len']
    EPOCHS = loaded_config['training']['epochs']
    FEATURE_DIM = REGION_SIZE * CHANNELS if SEQUENCE_ORDER == 'TR' else CHANNELS * SEQ_LEN
    NUM_CLASSES = loaded_config['training']['num_classes']

    NUM_HEADS = loaded_config['model']['num_heads']
    EMBED_DIM = loaded_config['model']['embed_dim']
    DEPTH = loaded_config['model']['depth']
    MLP_DIM = loaded_config['model']['mlp']
    DROP_RATE = loaded_config['model']['drop_rate']
    LEARNING_RATE = float(loaded_config['training']['learning_rate'])

    # Instantiate model
    model = VisionTransformer(
        FEATURE_DIM, SEQ_LEN, CHANNELS, NUM_CLASSES,
        EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE,
        REGION_SIZE, FEATURE_ARRANGEMENT, SEQUENCE_ORDER, 
        NORMALIZED_RGB,  POS_ENCODING_ENABLED, POOL_TYPE
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # # Training loop
    def train(model, loader, optimizer, criterion):
        model.train()
        total_loss, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()

        return total_loss / len(loader.dataset), correct / len(loader.dataset)


    EXPERIMENT_OUTPUT_PATH = create_folder_if_needed(PATH_OUTPUTFILE, EXPERIMENT_NAME)

    # @title
    # Training
    train_accuracies, test_accuracies = [], []
    # best_test_acc = 1 # Disables model saving
    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0

    overall_train_acc = 0
    overall_val_acc = 0
    overall_best_test_acc = 0

    final_train_acc = 0
    final_val_acc = 0
    final_test_acc = 0

    best_model_path = os.path.join(EXPERIMENT_OUTPUT_PATH, f"best_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{SEQUENCE_ORDER}.pth")
    best_model_test_path = os.path.join(EXPERIMENT_OUTPUT_PATH, f"best_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{SEQUENCE_ORDER}_test_based.pth")
    final_model_path = os.path.join(EXPERIMENT_OUTPUT_PATH, f"final_model_vit_{FEATURE_ARRANGEMENT}_{REGION_SIZE}_{SEQUENCE_ORDER}.pth")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Save the best model based on val acc
        if val_acc > best_val_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        if test_acc > overall_best_test_acc:
            overall_best_test_acc = test_acc
            overall_train_acc = train_acc
            overall_val_acc = val_acc
            torch.save(model.state_dict(), best_model_test_path)
            print(f"Best test model saved to {best_model_test_path}")

    final_train_acc = train_acc
    final_val_acc = val_acc
    final_test_acc = test_acc
    torch.save(model.state_dict(), final_model_path)
    print(f"Best test model saved to {final_model_path}")

    data_best_val_acc_based = {
        "test_acc": best_test_acc,
        "val_acc": best_val_acc,
        "train_acc": best_train_acc
    }

    data_overall_test_acc_based = {
        "test_acc": overall_best_test_acc,
        "val_acc": overall_val_acc,
        "train_acc": overall_train_acc
    }

    data_final_accs = {
        "test_acc": final_test_acc,
        "val_acc": final_val_acc,
        "train_acc": final_train_acc
    }

    save_metrics(EXPERIMENT_OUTPUT_PATH, data_final_accs, data_best_val_acc_based, data_overall_test_acc_based)

    # input_path = os.joininput_path
    # output_path = EXPERIMENT_OUTPUT_PATH

    # Load the best model weights
    model.load_state_dict(torch.load(best_model_test_path))
    model.eval() # Set the model to evaluation mode
    print(f"Loaded best model weights from {best_model_test_path}")

    folder_output_path = os.path.join(output_path, EXPERIMENT_NAME)
    generate_and_save_visualizations_itirapina(
        data, 
        dataset, 
        model, 
        test_loader, 
        device, 
        input_path, 
        folder_output_path,
        train_accuracies,
        test_accuracies,
        masks_dict 
    )