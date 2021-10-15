import argparse
import collections
import json
import sys
from typing import List
sys.path.append('MOCHA/')

import jsonlines
import tqdm
from allennlp.predictors.predictor import Predictor

from lerc.lerc_predictor import LERCPredictor


def get_lerc_score(
    lerc_metric: LERCPredictor,
    context: str,
    question: str,
    references: List[str],
    candidate: str
) -> float:
    # We accept multiple references. LERC score is w.r.t the highest rated reference.
    lerc_score = max([
        lerc_metric.predict_json({
            'context': context,
            'question': question,
            'reference': reference,
            'candidate': candidate
        })['pred_score']
        for reference in references
    ])

    # LERC is a regression model that returns a score (generally) between 1 and 5.
    # Squash the range to be between 0 and 1.
    lerc_score = (lerc_score - 1) / 4
    lerc_score = min(1, max(0, lerc_score))

    return lerc_score


def calculate_metrics(
    questions_file: str,
    answers_file: str,
    predictions_file: str,
    metrics_file: str
) -> None:
    lerc_metric = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz",
        "lerc"
    )
    questions = {d['id']: d for d in jsonlines.open(questions_file)}
    answers = {d['id']: d for d in jsonlines.open(answers_file)}
    predictions = {d['id']: d for d in jsonlines.open(predictions_file)}

    # Compute overall metric scores as well as per-dataset scores

    # Stores scores for different splits of the data
    metrics = collections.defaultdict(list)

    for query_id in tqdm.tqdm(questions):
        if query_id not in answers:
            raise ValueError(f"Entry in answer file not found for query {query_id}")

        if query_id not in predictions:
            print(f"Missing prediction for query {query_id}. Assigning score of 0.")
            lerc_score = 0
        else:
            lerc_score = get_lerc_score(
                lerc_metric,
                questions[query_id]['context'],
                questions[query_id]['question'],
                answers[query_id]['references'],
                predictions[query_id]['candidate']
            )

        metrics['avg_lerc'].append(lerc_score)

        # If there is a dataset key, we also compute per-dataset metrics
        if 'dataset' in questions[query_id]['metadata']:
            dataset = questions[query_id]['metadata']['dataset']
            metrics[f"{dataset}_lerc"].append(lerc_score)

    # Compute average metric scores
    metrics = {metric: sum(v)/len(v) for metric, v in metrics.items()}

    # Write the metric scores to file
    with open(metrics_file, "w") as f:
        f.write(json.dumps(metrics, indent=4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--questions_file",
        help="Filename of the questions to read. Expects a JSONL file with "
             "\"context\", \"question\", and \"id\" keys.",
        required=True
    )
    parser.add_argument(
        "-a", "--answers_file",
        help="Filename of the answers to read. Expects a JSONL file with \"id\" "
             "and \"references\" keys.",
        required=True
    )
    parser.add_argument(
        "-p", "--predictions_file",
        help="Filename of the predictions to read. Expects a JSONL file with \"id\" "
             "and \"candidate\" keys.",
        required=True
    )
    parser.add_argument(
        "-m", "--metrics_file",
        help="JSON file which the metrics of the predictions are writtent to.",
        required=True
    )
    args = parser.parse_args()
    calculate_metrics(
        args.questions_file,
        args.answers_file,
        args.predictions_file,
        args.metrics_file
    )


if __name__ == "__main__":
    main()
