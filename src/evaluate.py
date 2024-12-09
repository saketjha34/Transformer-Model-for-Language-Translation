import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from config import ModelConfig
from tqdm import tqdm
from transformer import build_transformer
from dataset import causal_mask
import random
from train import train_dataloader,val_dataloader
from torch.utils.data import Subset
import warnings
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_all_sentences(dataset:load_dataset, language:str):
    """
    Generator function that extracts all sentences in a specified language from a translation dataset.

    Args:
        dataset (load_dataset): A Hugging Face `load_dataset` object representing the dataset containing 
            translations in different languages.
        language (str): The target language for which sentences are to be extracted. This should correspond 
            to a key in the `translation` field of the dataset.

    Yields:
        str: Sentences in the specified language from the dataset.

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("opus_books", "en-fr")
        >>> language = "en"
        >>> sentences = get_all_sentences(dataset['train'], language)
        >>> for sentence in list(sentences)[:5]:
        ...     print(sentence)
    """
    
    for item in dataset:
        yield item['translation'][language]

def build_tokenizer(config, dataset:load_dataset, language:str) -> Tokenizer:
    """
    Builds or loads a tokenizer for a specified language using the Hugging Face `Tokenizers` library.

    Args:
        config (dict): A configuration dictionary containing the `tokenizer_file` key, which specifies 
            the file path template for saving/loading the tokenizer. The file path should include a 
            placeholder for the language.
        dataset (load_dataset): A Hugging Face `load_dataset` object representing the dataset containing 
            translations in different languages.
        language (str): The target language for which the tokenizer is being built or loaded. This should 
            correspond to a key in the `translation` field of the dataset.

    Returns:
        Tokenizer: A `Tokenizer` object built for the specified language.

    Example:
        >>> from tokenizers import Tokenizer
        >>> from datasets import load_dataset
        >>> config = {"tokenizer_file": "tokenizer_{language}.json"}
        >>> dataset = load_dataset("opus_books", "en-fr")
        >>> language = "en"
        >>> tokenizer = build_tokenizer(config, dataset['train'], language)
        >>> print(tokenizer.get_vocab_size())
    """
    
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Tokenizer language : {language} Build Complete.")
    
    return tokenizer

def greedy_decode(model: build_transformer, 
                  source: torch.Tensor, 
                  source_mask: torch.Tensor,
                  tokenizer_tgt: Tokenizer, 
                  max_len: int,
                  device: torch.device)->torch.Tensor:
    """
    Decodes a sequence from the source using a greedy decoding approach with a transformer model.

    Args:
        model ("build_transformer"): The transformer model to be used for encoding and decoding.
        source (torch.Tensor): The input tensor representing the source sequence. Shape: `(batch_size, seq_len)`.
        source_mask (torch.Tensor): A mask for the source input sequence. Shape: `(batch_size, 1, seq_len)`.
        tokenizer_tgt ("Tokenizer"): The tokenizer for the target language. Must provide `token_to_id` for special tokens.
        max_len (int): The maximum length for the decoded sequence.
        device (torch.device): The device (CPU or GPU) to perform the decoding on.

    Returns:
        torch.Tensor: A tensor representing the decoded sequence, excluding padding. Shape: `(seq_len,)`.

    Example:
        >>> from tokenizers import Tokenizer
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> src_tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        >>> src_mask = torch.tensor([[[1, 1, 1, 1]]], dtype=torch.int64)
        >>> tgt_tokenizer = Tokenizer.from_file("tokenizer_tgt.json")
        >>> transformer_model = build_transformer()  # Example transformer model
        >>> decoded_seq = greedy_decode(transformer_model, src_tensor, src_mask, tgt_tokenizer, max_len=20, device=device)
        >>> print(decoded_seq)
    """
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def translate_english_to_italian(model:build_transformer,
                                 tokenizer_src: Tokenizer,
                                 tokenizer_tgt: Tokenizer,
                                 english_sentence: str,
                                 max_len: int,
                                 device: torch.device) -> str:
    """
    Translates an English sentence into Italian using a trained transformer model.

    Args:
        model ("build_transformer"): The trained transformer model.
        tokenizer_src ("Tokenizer"): Tokenizer for the source language (English).
        tokenizer_tgt ("Tokenizer"): Tokenizer for the target language (Italian).
        english_sentence (str): The English sentence to translate.
        max_len (int): Maximum length for the translated sentence.
        device (torch.device): Device (CPU or GPU) to use for the translation.

    Returns:
        str: The translated sentence in Italian.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Tokenize the English sentence
        src_tokens = tokenizer_src.encode(english_sentence).ids
        src_tensor = torch.tensor([src_tokens], dtype=torch.int64).to(device)
        
        # Create a source mask
        src_mask = (src_tensor != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).to(device)

        # Perform greedy decoding to get the Italian translation
        translated_tokens = greedy_decode(
            model=model,
            source=src_tensor,
            source_mask=src_mask,
            tokenizer_tgt=tokenizer_tgt,
            max_len=max_len,
            device=device,
        )

        # Decode the translated tokens to get the Italian sentence
        italian_sentence = tokenizer_tgt.decode(translated_tokens.detach().cpu().numpy())
    
    return italian_sentence


def calculate_bleu_score(
                        model: build_transformer,
                        dataloader: torch.utils.data.DataLoader,
                        tokenizer_src: Tokenizer,
                        tokenizer_tgt: Tokenizer,
                        max_len: int,
                        device: torch.device) -> float:
    """
    Calculate the average BLEU score for a model on a given dataset with a progress bar.

    Args:
        model (callable): The trained transformer model.
        dataloader (torch.utils.data.DataLoader): Dataloader containing validation or test data.
        tokenizer_src (Tokenizer): Tokenizer for the source language (English).
        tokenizer_tgt (Tokenizer): Tokenizer for the target language (Italian).
        max_len (int): Maximum length for the generated sentences.
        device (torch.device): Device to run the model on.

    Returns:
        float: The average BLEU score over the dataset.
    """
    model.eval()
    total_bleu_score = 0.0
    count = 0

    # Initialize the progress bar
    with tqdm(total=len(dataloader), desc="Calculating BLEU Score", unit="batch") as pbar:
        with torch.no_grad():
            for batch in dataloader:
                encoder_input = batch["encoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)
                target_text = batch["tgt_text"]  # List of actual Italian sentences
                
                # Generate the translation using the model
                generated_tokens = translate_english_to_italian(
                    model=model,
                    tokenizer_src=tokenizer_src,
                    tokenizer_tgt=tokenizer_tgt,
                    english_sentence=batch["src_text"][0],  # Assuming batch size = 1
                    max_len=max_len,
                    device=device,
                )

                # Decode the generated tokens to text
                generated_text = generated_tokens.split()  # Split into words
                reference_text = [target_text[0].split()]  # Reference text as a list of words
                
                # Calculate BLEU score for this sentence
                bleu_score = sentence_bleu(reference_text, generated_text)
                total_bleu_score += bleu_score
                count += 1

                # Update the progress bar
                pbar.set_postfix({"BLEU (running avg)": f"{(total_bleu_score / count):.4f}"})
                pbar.update(1)

    # Calculate average BLEU score
    average_bleu_score = total_bleu_score / count if count > 0 else 0.0
    return average_bleu_score

if __name__ == "__main__":
    
    ## Laoding Pretrained Tokenizers from root directory
    config = ModelConfig().get_config()
    tokenizer_src = Tokenizer.from_file("../tokenizers/tokenizer_en.json")
    tokenizer_tgt = Tokenizer.from_file("../tokenizers/tokenizer_it.json")
    vocab_src_len = tokenizer_src.get_vocab_size()
    vocab_tgt_len = tokenizer_tgt.get_vocab_size()
    seq_len = 350

    ## Load Saved Model from save_path
    model = build_transformer(vocab_src_len, vocab_tgt_len, seq_len, seq_len,).to(device)
    model.load_state_dict(torch.load('../pytorch saved models/your_model.pth/', map_location=device))
    model.to(device)
    
    average_train_bleu = calculate_bleu_score(
    model=model,
    dataloader=train_dataloader,
    tokenizer_src=tokenizer_src,
    tokenizer_tgt=tokenizer_tgt,
    max_len=config['seq_len'],
    device=device,
    )

    average_val_bleu = calculate_bleu_score(
        model=model,
        dataloader=val_dataloader,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        max_len=config['seq_len'],
        device=device,
    )

    print(f"Train Dataset Average BLEU Score : {average_train_bleu:.7f}")
    print(f"Validation Dataset Average BLEU Score : {average_val_bleu:.7f}")