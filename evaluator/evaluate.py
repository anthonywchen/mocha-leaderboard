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


def get_lerc_scores(
    lerc_metric: LERCPredictor,
    contexts: List[str],
    questions: List[str],
    references: List[str],
    candidates: List[str]
) -> List[float]:
    assert len(contexts) == len(questions) == len(references) == len(candidates)
    batch_inputs = [
        {
            'context': contexts[i],
            'question': questions[i],
            'reference': references[i],
            'candidate': candidates[i]
        }
        for i in range(len(contexts))
    ]
    output_dicts = lerc_metric.predict_batch_json(batch_inputs)
    lerc_scores = [d['pred_score'] for d in output_dicts]

    # LERC is a regression model that returns a score (generally) between 1 and 5.
    # Squash the range to be between 0 and 1.
    lerc_scores = [min(1, max(0, (score-1)/4)) for score in lerc_scores]

    return lerc_scores


def calculate_metrics(
    questions_file: str,
    answers_file: str,
    predictions_file: str,
    metrics_file: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: int = -1
) -> None:
    lerc_metric = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/lerc-2020-11-18.tar.gz",
        "lerc",
        cuda_device=device
    )
    lerc_metric._dataset_reader.max_length = max_length
    questions = {d['id']: d for d in jsonlines.open(questions_file)}
    answers = {d['id']: d for d in jsonlines.open(answers_file)}
    predictions = {d['id']: d for d in jsonlines.open(predictions_file)}

    # Get scores for each reference for each query ID
    raw_metrics = collections.defaultdict(list)
    all_query_ids = list(questions.keys())
    for i in tqdm.tqdm(range(0, len(all_query_ids), batch_size)):
        batch_query_ids = []
        batch_contexts = []
        batch_questions = []
        batch_candidates = []
        batch_references = []

        # Construct batch inputs
        for query_id in all_query_ids[i:i+batch_size]:
            if query_id not in answers:
                raise ValueError(f"Entry in answer file not found for query {query_id}")
            elif query_id not in predictions:
                continue

            context = questions[query_id]['context']
            question = questions[query_id]['question']
            candidate = predictions[query_id]['candidate']

            for reference in answers[query_id]['references']:
                batch_query_ids.append(query_id)
                batch_contexts.append(context)
                batch_questions.append(question)
                batch_references.append(reference)
                batch_candidates.append(candidate)

        lerc_scores = get_lerc_scores(
            lerc_metric,
            batch_contexts,
            batch_questions,
            batch_references,
            batch_candidates
        )

        # Keep track of all scores for all references for each query
        for query_id, score in zip(batch_query_ids, lerc_scores):
            raw_metrics[query_id].append(score)

    # Compute overall metric scores as well as per-dataset score
    metrics = collections.defaultdict(list)
    for query_id in questions:
        lerc_score = 0 if query_id not in raw_metrics else max(raw_metrics[query_id])

        metrics['avg_lerc'].append(lerc_score)

        # If there is a dataset key, we also compute per-dataset metrics
        if 'dataset' in questions[query_id]['metadata']:
            dataset = questions[query_id]['metadata']['dataset']
            metrics[f"{dataset}_lerc"].append(lerc_score)

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
    parser.add_argument(
        "-b", "--batch_size",
        help="Batch size to do evaluation with. Default is 32.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--max_length",
        help="Max length of total input to LERC metric. Default is 512.",
        type=int,
        default=512
    )
    parser.add_argument(
        "-d", "--device",
        help="Device to run evaluation script on. Default is to run on CPU.",
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
        args.max_length,
        args.device
    )


if __name__ == "__main__":
    main()
