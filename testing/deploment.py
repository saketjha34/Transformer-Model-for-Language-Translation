import torch
from tokenizers import Tokenizer
from model import build_transformer
from utils.utils import greedy_decode, translate_english_to_italian, causal_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_src = Tokenizer.from_file("src_tokenizer_path")
tokenizer_tgt = Tokenizer.from_file("target_tokenizer_path")
vocab_src_len = tokenizer_src.get_vocab_size()
vocab_tgt_len = tokenizer_tgt.get_vocab_size()
seq_len = 350

model = build_transformer(vocab_src_len, vocab_tgt_len, seq_len, seq_len,).to(device)
model.load_state_dict(torch.load('path_to_your_pretrained_model', map_location=device))
model.to(device)

english_sentence = "How are you?"
italian_sentence = "Come stai?"
predicted_italian_sentence = translate_english_to_italian(
                                    model=model,
                                    tokenizer_src=tokenizer_src,
                                    tokenizer_tgt=tokenizer_tgt,
                                    english_sentence=english_sentence,
                                    max_len=350,
                                    device=device)

print(f"English Sentence : {english_sentence}")
print(f"Actual Italian Sentence : {italian_sentence}")
print(f"Translated Italian Sentence : {predicted_italian_sentence}")