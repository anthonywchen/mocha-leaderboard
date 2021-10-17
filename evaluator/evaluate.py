import argparse
import collections
import json
import sys
from typing import Dict
sys.path.append('MOCHA/')

import jsonlines
import tqdm
from allennlp.predictors.predictor import Predictor

from lerc.lerc_predictor import LERCPredictor


def get_lerc_scores(
    questions: Dict[str, dict],
    answers: Dict[str, dict],
    predictions: Dict[str, dict],
    batch_size: int = 32,
    device: int = -1
) -> Dict[str, float]:
    lerc_metric = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz",
        "lerc",
        cuda_device=device
    )

    # Get scores for each reference for each query ID
    raw_metrics = collections.defaultdict(list)

    all_query_ids = list(questions.keys())
    for i in tqdm.tqdm(range(0, len(all_query_ids), batch_size)):
        batch_query_ids = []
        batch_inputs = []

        # Construct batch inputs
        for query_id in all_query_ids[i:i+batch_size]:
            if query_id not in predictions:
                continue

            for reference in answers[query_id]['references']:
                batch_query_ids.append(query_id)
                batch_inputs.append({
                    'context': questions[query_id]['context'],
                    'question': questions[query_id]['question'],
                    'reference': reference,
                    'candidate': predictions[query_id]['candidate']
                })

        # Compute LERC scores for the batch
        output_dicts = lerc_metric.predict_batch_json(batch_inputs)
        lerc_scores = [d['pred_score'] for d in output_dicts]

        # LERC is a regression model that returns a score (generally) between 1 and 5.
        # Squash the range to be between 0 and 1.
        lerc_scores = [min(1, max(0, (score - 1) / 4)) for score in lerc_scores]

        # Keep track of all scores for all references for each query
        for query_id, score in zip(batch_query_ids, lerc_scores):
            raw_metrics[query_id].append(score)

    # Bin the LERC score by dataset
    metrics = collections.defaultdict(list)
    for query_id in questions:
        lerc_score = 0 if query_id not in raw_metrics else max(raw_metrics[query_id])
        dataset = questions[query_id]['metadata']['dataset']
        metrics[f"{dataset}_lerc"].append(lerc_score)

    # Compute average LERC score per dataset as well as macro-averaged LERC score
    metrics = {metric: sum(v)/len(v) for metric, v in metrics.items()}
    metrics['avg_lerc'] = sum(metrics.values())/len(metrics.values())
    return metrics


def calculate_metrics(
    questions_file: str,
    answers_file: str,
    predictions_file: str,
    metrics_file: str,
    batch_size: int = 32,
    device: int = -1
) -> None:
    questions = {d['id']: d for d in jsonlines.open(questions_file)}
    answers = {d['id']: d for d in jsonlines.open(answers_file)}
    predictions = {d['id']: d for d in jsonlines.open(predictions_file)}

    # Run check on data
    for query_id in questions:
        if query_id not in answers:
            raise ValueError(f"Entry in answer file not found for query {query_id}")
        elif query_id not in predictions:
            print(f"Missing prediction for query {query_id}. Assigning a score of 0.")

    # Compute metric scores for each metric
    metrics = {}
    metrics.update(get_lerc_scores(questions, answers, predictions, batch_size, device))

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
    parser.add_argument(
        "-b", "--batch_size",
        help="Batch size to do LERC evaluation with. Default is 32.",
        type=int,
        default=32
    )
    parser.add_argument(
        "-d", "--device",
        help="Device to run LERC evaluation on. Default is to run on CPU.",
        type=int,
        default=-1
    )
    args = parser.parse_args()
    calculate_metrics(
        args.questions_file,
        args.answers_file,
        args.predictions_file,
        args.metrics_file,
        args.batch_size,
        args.device
    )


if __name__ == "__main__":
    main()
