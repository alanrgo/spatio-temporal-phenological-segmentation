from data_loader.serra_do_cipo.loader import load_raw_data
import os
import argparse

from utils.vit_setup_loader import load_config

parser = argparse.ArgumentParser(description='Vit Experiments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str,
                    required=True,
                    help='serra_do_cipo, itirapinav1, ou itirapinav2')
parser.add_argument('--input_path', type=str,
                    required=True,
                    help='path/to/input/files')
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

# Read data
data = load_raw_data(input_path)
# Ler dicionario de classes de itirapina

path = os.path.join(setup_path, 'setup_common_training.yaml')
loaded_config = load_config(path)
print(loaded_config)

# X_train, Y_train, X_val, Y_val, X_test, Y_test = load_itirapina_data(data)
# train_loader, val_loader, test_loader  = sc_create_loaders(X_train, Y_train, X_val, Y_val, X_test, Y_test)
