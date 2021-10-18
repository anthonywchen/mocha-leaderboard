# MOCHA QA Evaluation

This repository contains all the information needed to run evaluation of MOCHA predictions using the LERC metric locally.
This can be useful if you want to run evaluation on the validation set which is available [here](../data).

### Setup
In order to run the evaluation script you will first need to install the requirements in the [requirements.txt](../requirements.txt) file.
You will then need to download the MOCHA repository [from this link](https://github.com/anthonywchen/mocha) into the root of this repository.

### Data 
Running evaluation requires three files:
* Questions file: A JSONLines file containing context and questions. Each line of this file requires four keys: an `id` key, a `context` key, a `question` key, and a `metadata` key.
* Answers file: A JSONLines file containing answers to the questions. Each line of this file requires two keys: an `id` key and a `references` key.
* Predictions file: A JSONLines file containing model predictions. Each line of this file requires two keys: an `id` key and a `candidate` key.

An example of each of these files is provided under the `dummy_*.jsonl` files.

Validation question and answer files are provided in the [`data/`](../data) directory. 
The test questions are also provided in this directory. 
We keep the test answers hidden and scores are provided when predictions are submitted to the leaderboard.

### Running Evaluation
Evaluation is handled through the `evaluate.py` script.
There are four required flags specifying paths to the question, answer, prediction, and an output metrics file and two optional flags specifying the device to run on and the batch size to do evaluation with (for LERC).

An example run of this script on some dummy data using the CPU is:
```bash
python evaluator/evaluate.py \
  --questions_file evaluator/dummy_questions.jsonl \
  --answers_file evaluator/dummy_answers.jsonl \
  --predictions_file evaluator/dummy_predictions.jsonl \
  --metrics_file evaluator/dummy_metrics.jsonl \
  --device -1
```

The metrics file contains scores for each constituent dataset with LERC, BLEU-1, and METEOR metrics, as well as a macro-averaged scores for each metric.
We choose a macro-average as each constituent dataset has a varying number of instances.