import torch
import torch.nn as nn
from transformers import ViTModel, BertLMHeadModel

class CrossAttention(nn.Module):
    """
    CrossAttention module to perform multi-head attention between text and image features.

    Args:
        hidden_size (int): The hidden size of the attention layers.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
    
    def forward(self, text_features, image_features):
        # Adjust dimensions: (seq_len, batch_size, hidden_size)
        text_features = text_features.permute(1, 0, 2)
        image_features = image_features.permute(1, 0, 2)
        
        attended_features, _ = self.attention(text_features, image_features, image_features)
        
        # Return to original dimension order
        return attended_features.permute(1, 0, 2)

class ImageCaptioningModel(nn.Module):
    """
    Image Captioning model that integrates a vision transformer (ViT) as the encoder
    and a BERT model as the decoder with an added cross-attention mechanism.

    Args:
        vit_model (ViTModel): Pretrained Vision Transformer model.
        decoder_model (BertLMHeadModel): Pretrained BERT language model.
    """

    def __init__(self, vit_model, decoder_model):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = vit_model
        self.decoder = decoder_model
        
        self.dim_reduction = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
        self.cross_attention = CrossAttention(self.decoder.config.hidden_size)

    def forward(self, images, captions, attention_mask):
        with torch.no_grad():
            image_features = self.encoder(images).last_hidden_state
        
        reduced_features = self.dim_reduction(image_features)
        
        text_embeds = self.decoder.bert.embeddings(input_ids=captions)
        attended_features = self.cross_attention(text_embeds, reduced_features)
        
        outputs = self.decoder(inputs_embeds=attended_features, attention_mask=attention_mask, labels=captions)
        return outputs.loss  

def generate_caption(self, image, tokenizer, max_length=64):
    """
    Generate caption for a given image using the trained model.

    Args:
        image (torch.Tensor): Input image tensor.
        tokenizer (BertTokenizer): Tokenizer for text processing.
        max_length (int): Maximum length of the generated caption.

    Returns:
        str: Generated caption.
    """

    with torch.no_grad():
        image_features = self.encoder(image).last_hidden_state
        reduced_features = self.dim_reduction(image_features)
        
        input_ids = torch.tensor([[tokenizer.cls_token_id]]).to(image.device)
        
        for _ in range(max_length):
            text_embeds = self.decoder.bert.embeddings(input_ids)
            attended_features = self.cross_attention(text_embeds, reduced_features)
            
            outputs = self.decoder(inputs_embeds=attended_features)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)