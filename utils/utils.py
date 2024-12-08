import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
import random
from torch.utils.data import Subset
from models.transformer import build_transformer
import warnings
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import math
import os


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

def causal_mask(size:int)->bool:
    """
    Creates a causal mask for autoregressive models.

    Args:
        size (int): The size of the square matrix for the mask.

    Returns:
        bool: A boolean tensor of shape (1, size, size), where the upper triangular part above the diagonal 
              is masked (False) and the rest is unmasked (True).

    Example:
        >>> import torch
        >>> mask = causal_mask(5)
        >>> print(mask)
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0



def get_random_subset(dataset:Dataset, fraction:float):
    """
    Get a random subset of a dataset.
    
    Args:
        dataset (Dataset): The original dataset.
        fraction (float): Fraction of the dataset to keep (0 < fraction <= 1).
    
    Returns:
        Subset: A smaller dataset containing the random subset.
    """
    assert 0 < fraction <= 1, "Fraction must be in the range (0, 1]."
    
    total_size = len(dataset)
    subset_size = int(total_size * fraction)
    random_indices = random.sample(range(total_size), subset_size)
    return Subset(dataset, random_indices)

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


def eval_model(model: build_tokenizer,
               greedy_decode: callable,
               val_dataloader: torch.utils.data.DataLoader, 
               tokenizer_src: Tokenizer, 
               tokenizer_tgt: Tokenizer,
               max_len: int, 
               device: torch.device, 
               num_examples:int=2) -> None:
    """
    Evaluates a transformer model by generating predictions on a validation dataset using greedy decoding.

    Args:
        model ("build_transformer"): The transformer model to be evaluated.
        greedy_decode (callable): The function for performing greedy decoding on the model.
        val_dataloader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        tokenizer_src ("Tokenizer"): Tokenizer for the source language.
        tokenizer_tgt ("Tokenizer"): Tokenizer for the target language.
        max_len (int): Maximum length for the decoded sequence.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.
        num_examples (int, optional): Number of examples to print during evaluation. Defaults to 2.

    Returns:
        None: Prints the source text, expected target text, and predicted output to the console.

    Example:
        >>> from tokenizers import Tokenizer
        >>> from torch.utils.data import DataLoader
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> val_loader = DataLoader(val_dataset, batch_size=1)
        >>> src_tokenizer = Tokenizer.from_file("tokenizer_src.json")
        >>> tgt_tokenizer = Tokenizer.from_file("tokenizer_tgt.json")
        >>> eval_model(transformer_model, greedy_decode, val_loader, src_tokenizer, tgt_tokenizer, max_len=50, device=device)
    """
    
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print('-'*console_width)
            print(f"{f'SOURCE: ':>12}{source_text}")
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print('-'*console_width)
                break

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

def train_model(model: build_tokenizer,
                greedy_decode: callable,
                tokenizer_src:Tokenizer,
                tokenizer_tgt:Tokenizer,
                val_dataloader: torch.utils.data.DataLoader,
                train_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim,
                loss_fn: nn.CrossEntropyLoss,
                english_sentence:str,
                config: dict, 
                device: torch.device) -> None:
    """
    Trains a transformer model for a sequence-to-sequence task using the provided dataset and configuration.

    Args:
        model ("build_transformer"): The transformer model to train.
        greedy_decode (callable): Function for greedy decoding during validation.
        tokenizer_src ("Tokenizer"): Tokenizer for the source language.
        tokenizer_tgt ("Tokenizer"): Tokenizer for the target language.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (nn.CrossEntropyLoss): Loss function for computing the training loss.
        config (dict): Configuration dictionary containing hyperparameters such as `num_epochs` and `seq_len`.
        device (torch.device): Device (CPU or GPU) to use for training.

    Returns:
        None: The function trains the model in-place and prints progress and evaluation results after each epoch.

    Example:
        >>> from tokenizers import Tokenizer
        >>> from torch.utils.data import DataLoader
        >>> import torch.nn as nn
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> config = {"num_epochs": 10, "seq_len": 50}
        >>> train_model(
        ...     model=transformer_model,
        ...     greedy_decode=greedy_decode,
        ...     tokenizer_src=src_tokenizer,
        ...     tokenizer_tgt=tgt_tokenizer,
        ...     val_dataloader=val_loader,
        ...     train_dataloader=train_loader,
        ...     optimizer=torch.optim.Adam(transformer_model.parameters(), lr=0.001),
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     config=config,
        ...     device=device,
        ... )
    """
    
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    initial_epoch = 0

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Run validation at the end of every epoch
        eval_model(model, greedy_decode ,val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        
        predicted_italian_sentence = translate_english_to_italian(
                model=model,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                english_sentence=english_sentence,
                max_len=50,
                device=device,
            )
        
        print()
        print("-"*150)
        print(f"English Sentence: {english_sentence}")
        print(f"Actual Italian Sentence: {'come stai amico mio?'}")
        print(f"Predicted Italian Sentence: {predicted_italian_sentence}")
        print()


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