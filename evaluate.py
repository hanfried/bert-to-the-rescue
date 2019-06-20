import sys

import click
import json_tricks as json
from sklearn.metrics import classification_report
import torch

from defaults import BATCH_SIZE, JSON_ARGS
from model import BertBinaryClassifier
from predict import predict


@click.command()
@click.option("-d", "--data-file", type=click.File("r"), default="data.json")
@click.option("-m", "--model-file", type=click.Path(readable=True), default="model.pt")
@click.option("-o", "--output-file", type=click.File("w"), default=sys.stdout)
def cli(data_file, model_file, output_file):
    data = json.load(data_file)
    model = BertBinaryClassifier().cuda()
    model.load_state_dict(torch.load(model_file))
    predictions, _ = predict(model, data["test_tokens_ids"], BATCH_SIZE)

    json.dump(
        classification_report(data["test_y"], predictions, output_dict=True),
        output_file,
        **JSON_ARGS
    )


if __name__ == "__main__":
    cli()
