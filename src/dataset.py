import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

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

class BilingualDataset(Dataset):
    """
    A custom PyTorch dataset class for bilingual translation tasks.

    Args:
        dataset (Dataset): A Hugging Face `Dataset` object containing translation data with `translation` fields.
        tokenizer_src (Tokenizer): Tokenizer for the source language.
        tokenizer_tgt (Tokenizer): Tokenizer for the target language.
        src_lang (str): Source language key in the `translation` field of the dataset.
        tgt_lang (str): Target language key in the `translation` field of the dataset.
        seq_len (int): Fixed sequence length for the input and output tensors.

    Returns:
        dict: A dictionary containing the following keys:
            - "encoder_input" (torch.Tensor): Tokenized and padded source sentence (seq_len).
            - "decoder_input" (torch.Tensor): Tokenized and padded target sentence with `<SOS>` (seq_len).
            - "encoder_mask" (torch.Tensor): Boolean mask for the encoder input (1, 1, seq_len).
            - "decoder_mask" (torch.Tensor): Boolean mask for the decoder input (1, seq_len, seq_len).
            - "label" (torch.Tensor): Tokenized and padded target sentence with `<EOS>` (seq_len).
            - "src_text" (str): Original source sentence.
            - "tgt_text" (str): Original target sentence.

    Example:
        >>> from tokenizers import Tokenizer
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("opus_books", "en-fr")
        >>> src_tokenizer = Tokenizer.from_file("tokenizer_en.json")
        >>> tgt_tokenizer = Tokenizer.from_file("tokenizer_fr.json")
        >>> bilingual_ds = BilingualDataset(
        ...     dataset=dataset['train'],
        ...     tokenizer_src=src_tokenizer,
        ...     tokenizer_tgt=tgt_tokenizer,
        ...     src_lang="en",
        ...     tgt_lang="fr",
        ...     seq_len=32
        ... )
        >>> sample = bilingual_ds[0]
        >>> print(sample["encoder_input"])
        >>> print(sample["decoder_input"])
        >>> print(sample["label"])
    """
    def __init__(self, dataset:Dataset, tokenizer_src:Tokenizer, tokenizer_tgt:Tokenizer, src_lang:str, tgt_lang:str, seq_len:int):
        super().__init__()
        self.seq_len = seq_len

        self.ds = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self)->int:
        return len(self.ds)

    def __getitem__(self, idx)->dict[str:torch.Tensor]:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }