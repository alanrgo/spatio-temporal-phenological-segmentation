# !/bin/bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
source activate convnet
python --version

pip install torch==1.7.1 torchvision==0.8.2
pip install pyyaml
pip3 install --upgrade pip
pip3 install packaging
pip install "numpy<2.0"
pip install "setuptools<60" "wheel<0.38"
pip install easydict
pip install neptune-contrib
pip install scikit-learn
pip install protobuf==3.20.0
pip install scikit-image
pip install scipy==1.1.0
pip install rasterio matplotlib
pip install tensorflow==2.15.0

# # Install CUDA and cuDNN via conda-forge for Python 3.9 compatibility
conda install -y -c conda-forge cudnn=8.9.2 cuda-toolkit=12.1

# Export paths for this session
export CONDA_PREFIX=$(python -c 'import os; print(os.environ.get("CONDA_PREFIX",""))')
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"

# Optional: also include common CUDA locations if present
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-11.8/lib64"

# Persist for future shells
echo 'export CONDA_PREFIX=$(python -c '"'"'import os; print(os.environ.get("CONDA_PREFIX",""))'"'"')' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify TensorFlow GPU availability
python - <<'PY'
import tensorflow as tf
print('TF:', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY


# conda install pytorch-cuda=12.5