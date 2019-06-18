import os
import random

import numpy as np
import torch


BERT_TOKENS_MAX = 512
SEED = int(os.environ.get("SEED", 321))


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
