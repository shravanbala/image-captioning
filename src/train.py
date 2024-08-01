import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import ImageCaptioningModel
from .dataset import FlickrDataset, load_flickr30k_dataset
from transformers import ViTFeatureExtractor, ViTModel, BertTokenizer, BertConfig, BertLMHeadModel
from .utils import get_transform

def train(model, train_loader, optimizer, device):
    """
    Train the Image Captioning model for one epoch.

    Args:
        model (torch.nn.Module): The image captioning model.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on.

    Returns:
        float: Average training loss over the epoch.
    """

    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        images, captions, attention_mask = [item.to(device) for item in batch]
        
        optimizer.zero_grad()
        loss = model(images, captions, attention_mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """
    Validate the Image Captioning model.

    Args:
        model (torch.nn.Module): The image captioning model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on.

    Returns:
        float: Average validation loss.
    """

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images, captions, attention_mask = [item.to(device) for item in batch]
            loss = model(images, captions, attention_mask)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(config):
    """
    Train and validate the Image Captioning model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing training parameters.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    train_data, val_data, _ = load_flickr30k_dataset(config['csv_file'], config['image_folder'])
    
    # Initialize tokenizer and feature extractor
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Create datasets and dataloaders
    train_dataset = FlickrDataset(train_data, tokenizer, feature_extractor, transform=get_transform())
    val_dataset = FlickrDataset(val_data, tokenizer, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    # Initialize the models
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', num_hidden_layers=config['vit_layers'])
    vit_model.eval()
    for param in vit_model.parameters():
        param.requires_grad = False
    
    decoder_config = BertConfig.from_pretrained('bert-base-uncased')
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder_config.num_hidden_layers = config['decoder_layers']
    decoder_config.hidden_size = config['hidden_size']
    decoder_config.intermediate_size = config['intermediate_size']
    decoder_config.num_attention_heads = config['num_attention_heads']
    decoder_model = BertLMHeadModel(decoder_config)
    
    # Create the full model
    model = ImageCaptioningModel(vit_model, decoder_model).to(device)
    
    # Initialize optimizer
    learning_rate = float(config['learning_rate'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), config['model_save_path'])
    print("Training completed and model saved.")

if __name__ == "__main__":
    from .utils import load_config
    config = load_config('configs/config.yaml')
    train_model(config)