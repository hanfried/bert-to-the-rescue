import json_tricks as json
from typing import List, Tuple
import random

import click
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from torchnlp.datasets import imdb_dataset

from defaults import BERT_MODEL, BERT_TOKENS_MAX, JSON_ARGS

Tokens = List[List[str]]
TokenIds = List[List[int]]


def tokenize(
    texts: List[str], bert_model: str = BERT_MODEL, do_lower_case: bool = True
) -> Tuple[Tokens, TokenIds]:
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    tokens = [["[CLS]"] + tokenizer.tokenize(t)[: BERT_TOKENS_MAX - 1] for t in texts]
    tokens_ids = pad_sequences(
        [tokenizer.convert_tokens_to_ids(t) for t in tokens],
        maxlen=BERT_TOKENS_MAX,
        truncating="post",
        padding="post",
        dtype="int",
    )

    return tokens, tokens_ids


def preprocess_imdb(train_size: int = 1000, test_size: int = 100) -> dict:
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

    train_tokens, train_tokens_ids = tokenize(train_texts)
    test_tokens, test_tokens_ids = tokenize(test_texts)

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


@click.command()
@click.option("--train-size", type=int, default=1000)
@click.option("--test-size", type=int, default=100)
@click.option("-o", "--output-file", type=click.File("w"))
def cli(train_size, test_size, output_file):
    result = preprocess_imdb(train_size=train_size, test_size=test_size)

    if output_file:
        json.dump(result, output_file, **JSON_ARGS)
    else:
        print(json.dumps(result, **JSON_ARGS))


if __name__ == "__main__":
    cli()
