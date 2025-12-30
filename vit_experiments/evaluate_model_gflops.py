from torchvision.models import resnet50
import torch
from thop import profile
from thop import clever_format

from data_loader.serra_do_cipo.loader import load_raw_data
import os
import argparse

from utils.vit_setup_loader import load_config
from models.vit import VisionTransformer
import torch

parser = argparse.ArgumentParser(description='Vit GFLOPs evaluation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str,
                    required=True,
                    help='serra_do_cipo, itirapinav1, ou itirapinav2')
parser.add_argument('--setup_path', type=str,
                    required=True,
                    help='path/to/setups')
args = parser.parse_args()

dataset = args.dataset
setup_path = args.setup_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
path = os.path.join(setup_path, 'setup_common_training.yaml')
loaded_config = load_config(path)
print(loaded_config)

# 1, 37, 9, 3 itirapina_v2 region 9
# 1, 37, 5, 3 itirapina_v2 region 5
# 1, 37, 1, 3 itirapina_v2 region 1

# 1, 13, 9, 3 serra_cipo region 9
# 1, 13, 5, 3 serra_cipo region 9
# 1, 13, 1, 3 serra_cipo region 9

setup_path = setup_path
setup_list = sorted(os.listdir(setup_path))
print(setup_list)

dict_ignore = {
    'setup_common_training.yaml': 1
}
print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")
for setup in setup_list:
    if setup in dict_ignore:
        continue
    setup_file_path = os.path.join(setup_path, setup)
    loaded_config = load_config(setup_file_path)
    EXPERIMENT_NAME = loaded_config['experiment_name']

    OUT_AS_BLACK = True
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
    )

    # Move model to CPU for FLOPs profiling (avoids CUBLAS issues)
    model = model.cpu()
    model.eval()

    # Create input tensor on CPU
    input = torch.randn((1, SEQ_LEN, 9, CHANNELS), device='cpu')
    # Profile FLOPs on CPU
    macs, params = profile(model, inputs=(input, ), verbose=False)
    
    print(
        "%s | %.2f | %.2f" % (f'VIT - {EXPERIMENT_NAME}', params / (1000 ** 2), macs / (1000 ** 3))
    )




