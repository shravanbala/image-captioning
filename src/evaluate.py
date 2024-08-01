import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
from rouge_score import rouge_scorer
import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from .model import ImageCaptioningModel
from .dataset import FlickrDataset, load_flickr30k_dataset
from transformers import ViTFeatureExtractor, ViTModel, BertTokenizer, BertConfig, BertLMHeadModel

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def generate_caption(model, image_tensor, tokenizer, config):
    """
    Generate a caption for a given image using the trained model.

    Args:
        model (ImageCaptioningModel): The trained image captioning model.
        image_tensor (torch.Tensor): Tensor representation of the image.
        tokenizer (BertTokenizer): Tokenizer for text processing.
        config (dict): Configuration dictionary containing generation parameters.

    Returns:
        str: Generated caption.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        image_features = model.encoder(image_tensor.unsqueeze(0)).last_hidden_state
        reduced_features = model.dim_reduction(image_features)
        
        input_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)
        
        for _ in range(config['max_length']):
            text_embeds = model.decoder.bert.embeddings(input_ids)
            attended_features = model.cross_attention(text_embeds, reduced_features)
            
            outputs = model.decoder(inputs_embeds=attended_features)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def evaluate(model, test_loader, tokenizer, config):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model (ImageCaptioningModel): The trained image captioning model.
        test_loader (DataLoader): DataLoader for the test dataset.
        tokenizer (BertTokenizer): Tokenizer for text processing.
        config (dict): Configuration dictionary containing evaluation parameters.

    Returns:
        dict: Evaluation scores (METEOR and ROUGE).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    generated_captions = []
    reference_captions = []
    
    for image_tensors, captions, _ in tqdm(test_loader, desc="Generating captions"):
        for image_tensor, caption in zip(image_tensors, captions):
            generated_caption = generate_caption(model, image_tensor.to(device), tokenizer, config)
            generated_captions.append(generated_caption)
            reference_captions.append([tokenizer.decode(caption, skip_special_tokens=True)])
    
    # METEOR Score
    meteor_scores = [
        meteor_score([word_tokenize(ref[0])], word_tokenize(hyp))
        for ref, hyp in zip(reference_captions, generated_captions)
    ]
    avg_meteor = np.mean(meteor_scores)
    
    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {metric: [] for metric in ['rouge1', 'rouge2', 'rougeL']}
    for ref, hyp in zip(reference_captions, generated_captions):
        scores = scorer.score(ref[0], hyp)
        for metric in rouge_scores:
            rouge_scores[metric].append(scores[metric].fmeasure)
    avg_rouge_scores = {metric: np.mean(scores) for metric, scores in rouge_scores.items()}


    return {
        'METEOR': avg_meteor,
        'ROUGE': avg_rouge_scores
    }

def evaluate_model(config):
    """
    Evaluate the image captioning model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing evaluation parameters.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    _, _, test_data = load_flickr30k_dataset(config['csv_file'], config['image_folder'])
    
    # Initialize tokenizer and feature extractor
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Create test dataset and dataloader
    test_dataset = FlickrDataset(test_data, tokenizer, feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    # Load the trained model
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', num_hidden_layers=config['vit_layers'])
    decoder_config = BertConfig.from_pretrained('bert-base-uncased')
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder_config.num_hidden_layers = config['decoder_layers']
    decoder_config.hidden_size = config['hidden_size']
    decoder_config.intermediate_size = config['intermediate_size']
    decoder_config.num_attention_heads = config['num_attention_heads']
    decoder_model = BertLMHeadModel(decoder_config)
    
    model = ImageCaptioningModel(vit_model, decoder_model).to(device)
    model.load_state_dict(torch.load(config['model_load_path'], map_location=device))
    
    # Evaluate the model
    scores = evaluate(model, test_loader, tokenizer, config)
    
    # Print results
    print("Evaluation Scores:")
    for metric, score in scores.items():
        if isinstance(score, dict):
            print(f"{metric}:")
            for sub_metric, sub_score in score.items():
                print(f"  {sub_metric}: {sub_score:.4f}")
        else:
            print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    from .utils import load_config
    config = load_config('configs/config.yaml')
    evaluate_model(config)
