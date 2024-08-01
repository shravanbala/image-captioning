import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

class FlickrDataset(Dataset):
    """
    Custom Dataset for loading Flickr30k dataset for image captioning.

    Args:
        data (list): List of tuples containing image paths and their respective captions.
        tokenizer (BertTokenizer): Tokenizer for text processing.
        feature_extractor (ViTFeatureExtractor): Feature extractor for image preprocessing.
        transform (callable, optional): Optional transform to be applied on an image.
    """

    def __init__(self, data, tokenizer, feature_extractor, transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image tensor, tokenized caption input IDs, and attention mask.
        """

        img_path, captions = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract image features
        image_tensor = self.feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze()
        
        # Tokenize the caption
        caption = self.tokenizer(captions[0], padding='max_length', max_length=64, truncation=True, return_tensors="pt")
        return image_tensor, caption['input_ids'].squeeze(), caption['attention_mask'].squeeze()


def load_flickr30k_dataset(csv_file, image_folder, test_size=0.1, val_size=0.1):
    """
    Load the Flickr30k dataset from a CSV file and split it into training, validation, and test sets.

    Args:
        csv_file (str): Path to the CSV file containing image names and captions.
        image_folder (str): Path to the folder containing images.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.

    Returns:
        tuple: Tuple containing training, validation, and test sets.
    """

    df = pd.read_csv(csv_file, delimiter='|', header=None, 
                     names=['image_name', 'comment_number', 'comment'])
    
    grouped = df.groupby('image_name')['comment'].apply(list).reset_index()

    # Create dataset list with image paths and comments
    dataset = []
    for _, row in grouped.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        if os.path.exists(image_path):
            dataset.append((image_path, row['comment']))

    # Split the dataset into train, validation, and test sets
    train_val, test = train_test_split(dataset, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    
    return train, val, test
