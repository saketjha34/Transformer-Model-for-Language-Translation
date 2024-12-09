---

# Transformer Model for English-to-Italian Translation

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [About the Dataset](#about-the-dataset)
4. [Usage](#usage)
5. [Results](#results)
6. [Transformer Model Architecture](#transformer-model-architecture)
7. [Deployment](#deployment)
8. [References](#references)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction
This project implements a Transformer model from scratch to perform English-to-Italian language translation. It uses PyTorch for model development and training, and leverages a dataset of bilingual texts to train the model. The project demonstrates the core concepts of neural machine translation and provides the flexibility to train a custom translator with any language pair.

## Project Structure
```plaintext
Transformer Model for Language Translation
│
├── pytorch pretrained model/
│   └── model.txt                   # Placeholder for the trained model file
├── src/
│   ├── config.py                   # Model and training configuration
│   ├── dataset.py                  # Dataset preparation utilities
│   ├── evaluate.py                 # Evaluation metrics (BLEU, METEOR, TER, etc.)
│   ├── train.py                    # Training script
│   ├── transformer.py              # Implementation of the Transformer model
├── testing/
│   ├── model.py                    # Testing model utilities
│   └── TestingNotebook.ipynb       # Jupyter Notebook for testing the model
├── tokenizers/
│   ├── tokenizer_en.json           # Tokenizer for English
│   └── tokenizer_it.json           # Tokenizer for Italian
├── utils/
│   ├── config.py                   # Utility configurations
│   └── utils.py                    # Additional helper functions
├── LICENSE                         # MIT License
├── ModelTrainingNotebook.ipynb     # Notebook for training the model
└── README.md                       # Project documentation
```

## About the Dataset
This project uses the [Opus Books Dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books), a collection of translated texts for various language pairs. The dataset contains high-quality, sentence-aligned bilingual data, making it ideal for machine translation tasks. You can customize the dataset variant and use other language pairs by modifying the `config.py` file.

## Usage

To train the Transformer Model on your custom Language pair.

1. Clone the repository:
    ```bash
    git clone https://github.com/saketjha34/Transformer-Model-for-Language-Translation
    cd Transformer-Model-for-Language-Translation
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configuration**: Customize the training and model parameters in `src/config.py`. Update the dataset variant and language pair as needed.
   
5. **Access the Configuration:**
   - Use the `get_config` method to retrieve the current configuration as a Python dictionary.
   ```python
   from config import ModelConfig
   config = ModelConfig().get_config()
   print(config)
   ```
   ### Dynamic Configuration During Training
   You can also dynamically modify the configuration during runtime:
   ```python
   config['num_epochs'] = 40  # Change number of epochs
   config['batch_size'] = 16  # Change batch size
   config['lang_src'] = 'fr'  # change source language
   config['lang_tgt'] = 'en'  # change target language
   config[seq_len'] = 450     # change max sequnce length
   ```
     
6. **Training**: Use `src/train.py` to train the model. Example:
   ```bash
   python src/train.py
   ```
   
7. **Evaluation**: Evaluate the model using the metrics implemented in `src/evaluate.py`.
   
8. **Testing**: Test the model on new sentences using `TestingNotebook.ipynb` in the `testing` folder.

### Explanation of the `src` Folder
- `config.py`: Contains all configurable parameters, such as batch size, learning rate, and dataset paths.
- `dataset.py`: Prepares the dataset for training and validation.
- `train.py`: The main script for training the model.
- `evaluate.py`: Calculates evaluation metrics (BLEU, TER, METEOR, etc.) to validate the model's performance.
- `transformer.py`: Core implementation of the Transformer architecture.

## Results
Below are the evaluation scores achieved during training and validation:

| Metric         | Training Score | Validation Score |
|----------------|----------------|-------------------|
| BLEU           | **41.5**       | **40.1**         |
| TER            | **58.0**       | **63.0**         |
| METEOR         | **0.47**       | **0.45**         |

## Transformer Model Architecture
Key features of the Transformer architecture:
- **Multi-Head Attention**: Captures relationships across all positions in a sequence.
- **Positional Encoding**: Represents the order of sequence elements.
- **Feed-Forward Networks**: Adds non-linearity to the model.
- **Encoder-Decoder Structure**: Encodes the input language and decodes it into the target language.
- **Layer Normalization**: Improves stability and accelerates training.

## Deployment
To test the model on custom English sentences, refer to the `TestingNotebook.ipynb` in the `testing` folder. It demonstrates how to load the trained model and perform translation with ease.

## References
1. **Transformer Paper**: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
2. **Opus Books Dataset**: [Hugging Face Dataset Page](https://huggingface.co/datasets/Helsinki-NLP/opus_books)

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request for review.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
