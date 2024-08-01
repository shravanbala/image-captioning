import yaml
from torchvision import transforms

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration parameters loaded from the YAML file.
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transform():
    """
    Get a composed transform with random augmentations for image preprocessing.
    
    Returns:
        torchvision.transforms.Compose: Composed transform with random horizontal flip,
                                         color jitter, and random rotation.
    """

    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
    ])
