import sys
from tqdm import tqdm
from typing import List, Tuple

import click
import json_tricks as json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from defaults import BATCH_SIZE, JSON_ARGS, device
from model import BertBinaryClassifier
from preprocessing import tokenize

Predictions = List[bool]
Logits = List[float]


def predict(
    model: BertBinaryClassifier, tokens_ids: np.ndarray, batch_size: int
) -> Tuple[Predictions, Logits]:
    tokens_tensor = torch.tensor(tokens_ids)
    masks = [[float(tid > 0) for tid in token_ids] for token_ids in tokens_ids]
    masks_tensor = torch.tensor(masks)

    dataset = TensorDataset(tokens_tensor, masks_tensor)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    model.eval()
    predictions: Predictions = []
    logits_all: Logits = []
    with torch.no_grad():
        steps_tqdm = tqdm(enumerate(dataloader), total=len(tokens_ids) // batch_size)
        for _, batch_data in steps_tqdm:
            token_ids, masks = tuple(bd.to(device) for bd in batch_data)
            logits = model(token_ids, masks).cpu().detach().numpy()
            predictions += list(logits[:, 0] > 0.5)
            logits_all += list(logits[:, 0])

    return predictions, logits_all


@click.command()
@click.option(
    "-t",
    "--texts-file",
    type=click.File("r"),
    default=sys.stdin,
    help="JSON list of strings for the texts",
)
@click.option("-m", "--model-file", type=click.Path(readable=True), default="model.pt")
@click.option("-o", "--output-file", type=click.File("w"), default=sys.stdout)
@click.option("-b", "--batch-size", type=int, default=BATCH_SIZE)
def cli(texts_file, model_file, output_file, batch_size):
    texts: List[str] = json.load(texts_file)
    _, tokens_ids = tokenize(texts)

    model = BertBinaryClassifier().cuda()
    model.load_state_dict(torch.load(model_file))
    predictions, logits = predict(model, tokens_ids, batch_size)

    json.dump({"predictions": predictions, "logits": logits}, output_file, **JSON_ARGS)


if __name__ == "__main__":
    cli()
