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

