import os
import random

from loguru import logger
import numpy as np
import torch

BATCH_SIZE = 4
BERT_MODEL = os.environ.get("BERT_MODEL", "bert-base-uncased")
BERT_TOKENS_MAX = 512
EPOCHS = 10
JSON_ARGS = {"indent": 4, "sort_keys": True}
SEED = int(os.environ.get("SEED", 321))


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("torch device={device}", device=device)
