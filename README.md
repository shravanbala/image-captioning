# Image Captioning Project

This project implements an image captioning model using a Vision Transformer (ViT) as the image encoder and a BERT-based model as the text decoder.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/shravanbala/image-captioning.git
   cd image-captioning
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the Flickr30k dataset and place the images in the `data/flickr30k_images/` directory.

4. Update the `configs/config.yaml` file with your specific settings.


## Usage

To train the model:
```bash
python main.py --mode train
```

To evaluate the model:
```bash
python main.py --mode evaluate
```

#### Evaluating the Model with Pretrained Checkpoint
1) Download the pretrained model checkpoint from the following link and place it in the `results/`  directory: [Pretrained Model Checkpoint](https://drive.google.com/file/d/1NNxWydu6kFrDwFgHceiDzxeg-uxtZNUY/view?usp=drive_link)

2) To evaluate the model using the pretrained checkpoint, run the following command:
 
``` bash
python main.py --mode evaluate 
```


## Project Structure

- `src/`: Contains the source code for the model, dataset, training, and evaluation.
- `configs/`: Contains configuration files.
- `data/`: Directory for storing the dataset (not tracked by git).
- `results/`: Directory for saving model checkpoints and evaluation results.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.