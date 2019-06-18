import json_tricks as json
import random

import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from torchnlp.datasets import imdb_dataset

from defaults import BERT_TOKENS_MAX


def preprocess_imdb(
    train_size: int = 1000,
    test_size: int = 100,
    bert_model: str = "bert-base-uncased",
    do_lower_case: bool = True,
):
    train_data, test_data = imdb_dataset(train=True, test=True)
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_data = train_data[:train_size]
    test_data = test_data[:test_size]

    train_texts, test_texts = (
        [d["text"] for d in data] for data in (train_data, test_data)
    )

    train_labels, test_labels = (
        [d["sentiment"] for d in data] for data in (train_data, test_data)
    )

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    train_tokens, test_tokens = (
        [["[CLS]"] + tokenizer.tokenize(t)[: BERT_TOKENS_MAX - 1] for t in texts]
        for texts in (train_texts, test_texts)
    )

    train_tokens_ids, test_tokens_ids = (
        [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        for tokens in (train_tokens, test_tokens)
    )

    train_y, test_y = (
        np.array(labels) == "pos" for labels in (train_labels, test_labels)
    )

    return {
        "test_labels": test_labels,
        "test_texts": test_texts,
        "test_tokens": test_tokens,
        "test_tokens_ids": test_tokens_ids,
        "test_y": test_y,
        "train_labels": train_labels,
        "train_texts": train_texts,
        "train_tokens": train_tokens,
        "train_tokens_ids": train_tokens_ids,
        "train_y": train_y,
    }


if __name__ == "__main__":
    result = preprocess_imdb(train_size=5, test_size=3)
    print(json.dumps(result, indent=4, sort_keys=True))