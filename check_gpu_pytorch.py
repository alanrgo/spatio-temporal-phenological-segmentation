import torch

print("PyTorch Version:", torch.__version__)
print("\nCUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

print("\n" + "="*50)
print("GPU Information:")
print("="*50)

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Reserved Memory: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # Test GPU
    print("\n" + "="*50)
    print("GPU Test:")
    print("="*50)
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("✓ GPU computation successful!")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
else:
    print("No CUDA-capable GPU found. CPU will be used.")
