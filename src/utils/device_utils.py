"""
Device Utilities for CPU/GPU Compatibility

Provides helper functions to ensure seamless operation on both CPU and GPU.
"""

import torch


def get_device(prefer_gpu=True, device_id=0):
    """
    Get the appropriate device (CPU or GPU) for tensor operations.
    
    Args:
        prefer_gpu (bool): If True, use GPU if available, otherwise CPU
        device_id (int): GPU device ID to use if multiple GPUs are available
    
    Returns:
        torch.device: Device object for tensor operations
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        if prefer_gpu and not torch.cuda.is_available():
            print("GPU requested but not available. Using CPU instead.")
        else:
            print("Using CPU")
    
    return device


def move_to_device(data, device):
    """
    Move data to the specified device.
    Handles tensors, lists, tuples, and dicts recursively.
    
    Args:
        data: Data to move (tensor, list, tuple, or dict)
        device (torch.device): Target device
    
    Returns:
        Data moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data


def to_numpy(tensor):
    """
    Convert a PyTorch tensor to numpy array, handling GPU tensors.
    
    Args:
        tensor (torch.Tensor): Input tensor
    
    Returns:
        numpy.ndarray: Numpy array
    """
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def get_memory_usage(device):
    """
    Get current GPU memory usage if using CUDA.
    
    Args:
        device (torch.device): Device to check
    
    Returns:
        dict: Memory usage statistics (allocated, reserved, free)
    """
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
        reserved = torch.cuda.memory_reserved(device) / 1e9    # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        free = total - allocated
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': free
        }
    else:
        return {'message': 'CPU mode - no GPU memory tracking'}


def clear_cuda_cache(device):
    """
    Clear CUDA cache if using GPU.
    
    Args:
        device (torch.device): Device to clear cache for
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    else:
        print("CPU mode - no cache to clear")


def print_device_info():
    """
    Print comprehensive device information.
    """
    print("="*60)
    print("Device Information")
    print("="*60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
    else:
        print("No GPU available. Using CPU only.")
    print("="*60)


def set_cuda_deterministic(deterministic=True):
    """
    Set CUDA operations to be deterministic for reproducibility.
    
    Args:
        deterministic (bool): If True, use deterministic algorithms
    
    Note:
        This may reduce performance but ensures reproducible results.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if deterministic:
            print("CUDA operations set to deterministic mode (reproducible but slower)")
        else:
            print("CUDA operations set to non-deterministic mode (faster but not reproducible)")


def enable_tf32(enable=True):
    """
    Enable or disable TF32 on Ampere GPUs for better performance.
    
    Args:
        enable (bool): If True, enable TF32
    
    Note:
        TF32 provides better performance on Ampere GPUs (RTX 30xx, A100)
        with minimal accuracy impact for most deep learning tasks.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable
        torch.backends.cudnn.allow_tf32 = enable
        if enable:
            print("TF32 enabled for Ampere GPUs (faster computation)")
        else:
            print("TF32 disabled (more precise but slower on Ampere GPUs)")


# Context manager for device handling
class DeviceContext:
    """
    Context manager for consistent device handling.
    
    Usage:
        with DeviceContext(prefer_gpu=True) as device:
            model = MyModel().to(device)
            data = data.to(device)
    """
    def __init__(self, prefer_gpu=True, device_id=0):
        self.prefer_gpu = prefer_gpu
        self.device_id = device_id
        self.device = None
    
    def __enter__(self):
        self.device = get_device(self.prefer_gpu, self.device_id)
        return self.device
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'cuda':
            clear_cuda_cache(self.device)


if __name__ == "__main__":
    # Test device utilities
    print_device_info()
    
    device = get_device(prefer_gpu=True)
    print(f"\nSelected device: {device}")
    
    # Test tensor movement
    test_tensor = torch.randn(2, 3, 4)
    test_tensor = move_to_device(test_tensor, device)
    print(f"\nTest tensor device: {test_tensor.device}")
    
    # Test memory info
    if device.type == 'cuda':
        memory_info = get_memory_usage(device)
        print(f"\nGPU Memory Usage:")
        for key, value in memory_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
