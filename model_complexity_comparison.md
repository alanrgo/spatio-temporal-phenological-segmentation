# Model Complexity Comparison: ConvNet vs ViT

## ConvNet25Temporal Results

| Dataset         | Num Inputs | Classes | Params(M) | FLOPs(G) |
|-----------------|------------|---------|-----------|----------|
| itirapina_v2    | 37         | 6       | 7.95      | 1.03     |
| serra_do_cipo   | 13         | 4       | 3.84      | 0.36     |

## Vision Transformer (ViT) Results for Comparison

### Itirapina v2:
- TR order (region 5 or 9): **1.59M params, 0.06 GFLOPs**
- RT order (region 5): **1.60M params, 0.01 GFLOPs**
- RT order (region 9): **1.60M params, 0.02 GFLOPs**
- RT order (region 1): **1.60M params, 0.00 GFLOPs**

### Serra do Cipó:
- TR order (all regions): **1.59M params, 0.02 GFLOPs**
- RT order (region 5): **1.60M params, 0.01 GFLOPs**
- RT order (region 9): **1.60M params, 0.02 GFLOPs**
- RT order (region 1): **1.60M params, 0.00 GFLOPs**

## Key Insights

### Parameters:
- **ConvNet is 2.4-5x larger in parameters**:
  - Itirapina v2: ConvNet 7.95M vs ViT 1.59-1.60M (~5x)
  - Serra do Cipó: ConvNet 3.84M vs ViT 1.59-1.60M (~2.4x)

### FLOPs:
- **ConvNet requires significantly more computation**:
  - Itirapina v2: ConvNet 1.03G vs ViT 0.00-0.06G (~17-1000x)
  - Serra do Cipó: ConvNet 0.36G vs ViT 0.00-0.02G (~18-360x)

### Efficiency Winner: **Vision Transformer (ViT)**
- ViT models are much more parameter-efficient
- ViT models require substantially fewer FLOPs
- ViT-RT with small regions (1-5) offers best computational efficiency

### ConvNet Characteristics:
- Parameters and FLOPs scale linearly with number of temporal inputs
- Each input day adds ~64 feature channels after initial convolution
- Heavy computation in early convolution layers processing full temporal sequences
- Two large FC layers (1024x1024 and 1024xClasses) contribute significantly to parameters

### ViT Characteristics:
- Fixed parameters regardless of region size (only positional encoding changes)
- FLOPs scale with number of tokens (sequence_length or region_size)
- RT order with small regions achieves minimal computation
- More efficient attention mechanism compared to ConvNet's dense convolutions

## Recommendation:
For deployment scenarios requiring:
- **Low latency/computation**: Use ViT-RT with region=1 (~100-1000x faster)
- **Parameter efficiency**: Any ViT variant (~2.4-5x fewer parameters)
- **Balanced performance**: ViT-TR or ViT-RT with region=5
