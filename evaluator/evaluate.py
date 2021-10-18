import argparse
import collections
import json
import sys
from typing import Dict
sys.path.append('MOCHA/')

import datasets
import jsonlines
import tqdm
from allennlp.predictors.predictor import Predictor
from nltk import word_tokenize as tokenize

from lerc.lerc_predictor import LERCPredictor


def aggregate_raw_metrics(
    questions: Dict[str, dict],
    raw_metrics: Dict[str, float],
    metric_name: str
) -> Dict[str, float]:
    # Bin the raw scores for the queries by dataset
    metrics = collections.defaultdict(list)
    for query_id in questions:
        dataset = questions[query_id]['metadata']['dataset']
        score = raw_metrics.get(query_id, 0)
        metrics[f"{dataset}_{metric_name}"].append(score)

    # Compute per dataset score as well as macro-averaged score over all datasets
    metrics = {dataset: sum(v)/len(v) for dataset, v in metrics.items()}
    metrics[f"avg_{metric_name}"] = sum(metrics.values())/len(metrics.values())
    return metrics


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
    for i in tqdm.tqdm(range(0, len(all_query_ids), batch_size), desc="LERC"):
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

    # Take the max score over all references for each query
    raw_metrics = {query_id: max(raw_metrics[query_id]) for query_id in raw_metrics}

    metrics = aggregate_raw_metrics(questions, raw_metrics, "lerc")
    return metrics


def get_bleu_scores(
    questions: Dict[str, dict],
    answers: Dict[str, dict],
    predictions: Dict[str, dict],
) -> Dict[str, float]:
    bleu_metric = datasets.load_metric("bleu")
    raw_metrics = {}
    for query_id in tqdm.tqdm(list(questions.keys()), desc="BLEU"):
        if query_id not in predictions:
            continue

        # Tokenize and lower-case before computing BLEU-1 metric.
        candidate = [tokenize(predictions[query_id]['candidate'].lower())]
        references = [[tokenize(r.lower()) for r in answers[query_id]['references']]]
        output_dict = bleu_metric.compute(
            predictions=candidate,
            references=references,
            max_order=1
        )
        raw_metrics[query_id] = output_dict['bleu']

    metrics = aggregate_raw_metrics(questions, raw_metrics, "bleu1")
    return metrics


def get_meteor_scores(
    questions: Dict[str, dict],
    answers: Dict[str, dict],
    predictions: Dict[str, dict],
) -> Dict[str, float]:
    meteor_metric = datasets.load_metric("meteor")
    raw_metrics = {}
    for query_id in tqdm.tqdm(list(questions.keys()), desc="METEOR"):
        if query_id not in predictions:
            continue

        # Get METEOR score for each reference and take max
        candidate = [predictions[query_id]['candidate']]
        meteor_scores = [
            meteor_metric.compute(predictions=candidate, references=[ref])['meteor']
            for ref in answers[query_id]['references']
        ]
        raw_metrics[query_id] = max(meteor_scores)

    metrics = aggregate_raw_metrics(questions, raw_metrics, "meteor")
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

    for query_id in questions:
        if query_id not in answers:
            raise ValueError(f"Entry in answer file not found for query {query_id}")
        elif query_id not in predictions:
            print(f"Missing prediction for query {query_id}. Assigning a score of 0.")

    metrics = {}
    metrics.update(get_lerc_scores(questions, answers, predictions, batch_size, device))
    metrics.update(get_bleu_scores(questions, answers, predictions))
    metrics.update(get_meteor_scores(questions, answers, predictions))

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
