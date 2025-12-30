# Spatio-Temporal Vegetation Segmentation By Using Convolutional Networks

<p align="center">
<figure>
  <img src="datasets.png" alt="Obtained Results" width="500">
  <figcaption>Figure 1. Examples of phenological images exploited in this work.</figcaption>
</figure> 
</p>

Plant phenology studies rely on long-term monitoring of life cycles of plants.
High-resolution unmanned aerial vehicles (UAVs) and near-surface technologies have been used for plant monitoring, demanding the creation of methods capable of locating and identifying plant species through time and space.
However, this is a challenging task given the high volume of data, the constant data missing from temporal dataset, the heterogeneity of temporal profiles, the variety of plant visual patterns, and the unclear definition of individuals’ boundaries in plant communities.
In this work, we propose a novel method, suitable for phenological monitoring, based on Convolutional Networks (ConvNets) to perform spatio-temporal vegetation pixel-classification on high resolution images.
We conducted a systematic evaluation using high-resolution vegetation image datasets associated with the Brazilian Cerrado biome.
Experimental results show that the proposed approach is effective, overcoming other spatio-temporal pixel-classification strategies.

### Citing

If you use this code in your research, please consider citing:

    @article{nogueiraGRSL2019spatio,
        author = {Keiller Nogueira and Jefersson A. dos Santos and Nathalia Menini and Thiago S. F. Silva and Leonor Patricia Morellato and Ricardo da S. Torres}
        title = {Spatio-Temporal Vegetation Pixel Classification By Using Convolutional Networks},
        journal = {{IEEE} Geoscience and Remote Sensing Letters},
        year = {2019},
        publisher={IEEE}
    }

### Setup 

`bash setup.sh`

### Results 

Itirapina v2

```
Iter 19900 -- Time 21:22:38.204014 -- Training Minibatch: Loss= 0.031165 Absolut Right Pred= 100 Overall Accuracy= 1.0000 Normalized Accuracy= 1.0000 Confusion Matrix= [[ 5  0  0  0  0  0] [ 0 34  0  0  0  0] [ 0  0  7  0  0  0] [ 0  0  0 28  0  0] [ 0  0  0  0 22  0] [ 0  0  0  0  0  4]]
Iter 19950 -- Time 21:24:28.055059 -- Training Minibatch: Loss= 0.030486 Absolut Right Pred= 100 Overall Accuracy= 1.0000 Normalized Accuracy= 1.0000 Confusion Matrix= [[ 1  0  0  0  0  0] [ 0 41  0  0  0  0] [ 0  0  4  0  0  0] [ 0  0  0 31  0  0] [ 0  0  0  0 21  0] [ 0  0  0  0  0  2]]
Iter 20000 -- Time 21:26:17.757911 -- Training Minibatch: Loss= 0.027842 Absolut Right Pred= 100 Overall Accuracy= 1.0000 Normalized Accuracy= 1.0000 Confusion Matrix= [[ 6  0  0  0  0  0] [ 0 32  0  0  0  0] [ 0  0  3  0  0  0] [ 0  0  0 36  0  0] [ 0  0  0  0 18  0] [ 0  0  0  0  0  5]]
-- Iter 20000 -- Training Epoch: Overall Accuracy= 0.998250 Normalized Accuracy= 0.997008 Confusion Matrix= [[ 3753     9     0     0     7     0] [   10 30625     3    27     1    11] [    0     6  4193     9     1     0] [    0    30     9 36703     6     2] [    9     6     0     5 21422     6] [    0    13     0     2     3  3129]]
---- Iter 20000 -- Validate: Overall Accuracy= 1582 Overall Accuracy= 0.078840 Normalized Accuracy= 0.166667 Confusion Matrix= [[   0    0    0 2523    0    0] [   0    0    0 7405    0    0] [   0    0    0 1527    0    0] [   0    0    0 1582    0    0] [   0    0    0 4028    0    0] [   0    0    0 3001    0    0]]
Iter 20050 -- Time 21:35:12.056335 -- Training Minibatch: Loss= 0.026534 Absolut Right Pred= 100 Overall Accuracy= 1.0000 Normalized Accuracy= 1.0000 Confusion Matrix= [[ 1  0  0  0  0  0] [ 0 33  0  0  0  0] [ 0  0  7  0  0  0] [ 0  0  0 36  0  0] [ 0  0  0  0 19  0] [ 0  0  0  0  0  4]]
Iter 20100 -- Time 21:37:01.896605 -- Training Minibatch: Loss= 0.028356 Absolut Right Pred= 100 Overall Accuracy= 1.0000 Normalized Accuracy= 1.0000 Confusion Matrix= [[ 5  0  0  0  0  0] [ 0 33  0  0  0  0] [ 0  0  3  0  0  0] [ 0  0  0 36  0  0] [ 0  0  0  0 18  0] [ 0  0  0  0  0  5]]
Iter 20150 -- Time 21:38:51.271946 -- Training Minibatch: Loss= 0.025895 Absolut Right Pred= 100 Overall Accuracy= 1.0000 Normalized Accuracy= 0.8333 Confusion Matrix= [[ 4  0  0  0  0  0] [ 0 33  0  0  0  0] [ 0  0  7  0  0  0] [ 0  0  0 34  0  0] [ 0  0  0  0 22  0] [ 0  0  0  0  0  0]]
```

```
#!/bin/bash

ativo (Corrigindo o erro anterior)
if [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate"
else
    echo "ERRO: Miniconda não encontrado. Verifique a instalação."
    exit 1
fi
```

# 2. Inicializa o conda para garantir (evita erro de CommandNotFound)
conda init bash

# 3. Remove ambiente antigo se existir (para começar limpo)
conda env remove -n convnet_legacy -y 2>/dev/null

# 4. Cria ambiente com Python 3.7 
# (Python 3.7 é OBRIGATÓRIO para usar scipy 1.1.0 e TF 1.15)
echo "--- Criando ambiente Python 3.7 ---"
conda create -n convnet_legacy python=3.7 -y

# 5. Ativa o ambiente
source activate convnet_legacy

# 6. Instalação via CONDA (Mais seguro para versões antigas)
# Instalamos o tensorflow-gpu 1.15 do canal oficial ou nvidia
echo "--- Instalando TensorFlow 1.15 e CUDA 10 ---"
# O cudatoolkit 10.0 é necessário para o TF 1.15
conda install -y tensorflow-gpu=1.15 cudatoolkit=10.0 -c anaconda

# 7. Instala as dependências exatas do seu código via PIP
echo "--- Instalando bibliotecas auxiliares antigas ---"
pip install "numpy<1.19"
# O scipy 1.1.0 é CRÍTICO para o seu código (tem o scipy.misc.imread)
pip install "scipy==1.1.0"
pip install "scikit-image<0.16"
pip install "pillow<7.0.0"
pip install scikit-learn==0.24.2
pip install rasterio matplotlib easydict pyyaml protobuf==3.20.0

echo "--- Verificando Instalação ---"
python -c "import tensorflow as tf; print('Versão TF:', tf.__version__); print('GPU Disponível:', tf.test.is_gpu_available())"