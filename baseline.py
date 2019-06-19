from typing import List, Union

import click
import json_tricks as json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report


def baseline_model(train_texts: List[str], train_labels: List[str]) -> Pipeline:
    return make_pipeline(CountVectorizer(ngram_range=(1, 3)), LogisticRegression()).fit(
        train_texts, train_labels
    )


def evaluate(
    model: Pipeline,
    test_texts: List[str],
    test_labels: List[str],
    output_dict: bool = False,
) -> Union[str, dict]:
    predictions = model.predict(test_texts)
    return classification_report(test_labels, predictions, output_dict=output_dict)


@click.command()
@click.option("-d", "--data-file", type=click.File("r"), default="data.json")
@click.option("-o", "--output-file", type=click.File("w"))
def cli(data_file, output_file):
    data = json.loads(data_file.read())
    model = baseline_model(data["train_texts"], data["train_labels"])
    args = (model, data["test_texts"], data["test_labels"])
    if output_file:
        json.dump(evaluate(*args, output_dict=True), output_file)
    else:
        print(evaluate(*args))


if __name__ == "__main__":
    cli()
