import torch
from thop import profile, clever_format
from convnet_pytorch_model import ConvNet25Temporal


def evaluate_convnet_flops():
    """Evaluate FLOPs and parameters for ConvNet25Temporal model"""
    
    # Configuration
    crop_size = 25
    num_timestamps = 13
    channels = 3
    batch_size = 1
    
    # Different dataset configurations
    datasets = {
        'itirapina_v2': {'num_inputs': 37, 'num_classes': 6},
        'serra_do_cipo': {'num_inputs': 13, 'num_classes': 4}
    }
    
    print("=" * 80)
    print("ConvNet25Temporal FLOPs and Parameters Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Crop Size: {crop_size}x{crop_size}")
    print(f"  Timestamps: {num_timestamps}")
    print(f"  Channels: {channels}")
    print(f"  Input feature dimension per patch: {crop_size}x{crop_size}x{num_timestamps*channels}")
    print("\n" + "-" * 80)
    
    print(f"\n{'Dataset':<20} {'Num Inputs':<12} {'Classes':<10} {'Params(M)':<15} {'FLOPs(G)':<15}")
    print("-" * 80)
    
    for dataset_name, config in datasets.items():
        num_inputs = config['num_inputs']
        num_classes = config['num_classes']
        
        # Create model
        model = ConvNet25Temporal(
            crop_size=crop_size,
            num_inputs=num_inputs,
            num_classes=num_classes,
            num_timestamps=num_timestamps,
            channels=channels
        )
        model.eval()
        
        # Create dummy input - list of tensors for each temporal input
        # Each tensor: [batch, crop_size, crop_size, num_timestamps * channels]
        inputs = tuple([
            torch.randn(batch_size, crop_size, crop_size, num_timestamps * channels)
            for _ in range(num_inputs)
        ])
        
        # Profile FLOPs and parameters
        macs, params = profile(model, inputs=(inputs,), verbose=False)
        
        # Convert to readable format
        macs_g = macs / (1000 ** 3)  # Convert to GFLOPs
        params_m = params / (1000 ** 2)  # Convert to millions
        
        print(f"{dataset_name:<20} {num_inputs:<12} {num_classes:<10} {params_m:<15.2f} {macs_g:<15.2f}")
    
    print("-" * 80)
    print("\nNotes:")
    print("  - Each 'input' represents one day of temporal data (13 timestamps)")
    print("  - Itirapina v2 has 37 days, Serra do Cipó has 13 days")
    print("  - FLOPs scale linearly with number of temporal inputs")
    print("  - Parameters scale linearly with number of input channels (num_inputs * 64)")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_convnet_flops()
