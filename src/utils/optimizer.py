"""
Optimizer configuration utilities for training.
"""

import torch.optim as optim


def get_optimizer(model, config):
    """
    Initialize optimizer based on configuration.
    
    Args:
        model: The model whose parameters will be optimized
        config: Dictionary containing optimizer configuration with keys:
                - name: 'SGD' or 'Adam'
                - lr: learning rate
                - For SGD: momentum, weight_decay, nesterov
                - For Adam: betas, eps, weight_decay
    
    Returns:
        optimizer: Initialized optimizer (SGD or Adam)
    """
    optimizer_name = config.get('name', 'Adam').upper()
    lr = config.get('lr', 0.001)
    
    if optimizer_name == 'SGD':
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0)
        nesterov = config.get('nesterov', False)
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        print(f"Optimizer: SGD (lr={lr}, momentum={momentum}, nesterov={nesterov})")
        
    elif optimizer_name == 'ADAM':
        betas = tuple(config.get('betas', [0.9, 0.999]))
        eps = config.get('eps', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        print(f"Optimizer: Adam (lr={lr}, betas={betas})")
        
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'SGD' or 'Adam'.")
    
    return optimizer
