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
import warnings
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import math
import os
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")