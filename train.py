from tqdm import tqdm

import click
import json_tricks as json
from loguru import logger
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from model import BertBinaryClassifier
from defaults import BATCH_SIZE, EPOCHS, device


def train(
    tokens_ids: np.ndarray, y: np.ndarray, batch_size: int, epochs: int
) -> BertBinaryClassifier:
    tokens_tensor = torch.tensor(tokens_ids)
    y_tensor = torch.tensor(y.reshape(-1, 1)).float()
    masks = [[float(tid > 0) for tid in token_ids] for token_ids in tokens_ids]
    masks_tensor = torch.tensor(masks)

    dataset = TensorDataset(tokens_tensor, masks_tensor, y_tensor)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    model = BertBinaryClassifier().cuda()
    optimizer = Adam(model.parameters(), lr=3e-6)

    torch.cuda.empty_cache()

    model.train()
    logger.debug(
        "cuda allocated memory {mem}M", mem=torch.cuda.memory_allocated(device) / 1e6
    )
    for epoch_nr in range(epochs):
        logger.info("epoch={nr}", nr=epoch_nr)
        loss = 0
        steps_tqdm = tqdm(enumerate(dataloader), total=len(y) // batch_size)
        for step_nr, batch_data in steps_tqdm:
            token_ids, masks, labels = tuple(bd.to(device) for bd in batch_data)
            logits = model(token_ids, masks)

            loss_func = torch.nn.BCELoss()
            batch_loss = loss_func(logits, labels)
            loss += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()

            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            steps_tqdm.set_postfix(loss=loss / (step_nr + 1))

    return model


@click.command()
@click.option("-d", "--data-file", type=click.File("r"), default="data.json")
@click.option("-b", "--batch-size", type=int, default=BATCH_SIZE)
@click.option("-e", "--epochs", type=int, default=EPOCHS)
@click.option(
    "-o", "--output-model", type=click.Path(writable=True), default="model.pt"
)
def cli(data_file, batch_size, epochs, output_model):
    data = json.loads(data_file.read())
    model = train(data["train_tokens_ids"], data["train_y"], batch_size, epochs)
    torch.save(model.state_dict(), output_model)


if __name__ == "__main__":
    cli()
