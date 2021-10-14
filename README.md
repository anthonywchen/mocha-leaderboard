# MOCHA Question Answering Leaderboard

This leaderboard is for evaluating generative question answering (QA) systems' predictions on a variety of generative QA datasets. 
We use a learned metric that has been trained to do generative QA evaluation known as **LERC** *(**L**earned **E**valuation for **R**eading **C**omprehension)* to do our evaluation.
The core datasets that we evaluate model predictions on are the four generative QA datasets that make up MOCHA.

**Note**: This leaderboard is not for evaluating question answering metrics. 
Rather, it uses the LERC metric that is trained on the MOCHA dataset to evaluate generative QA predictions.

### Setup
This repository has been tested with Python 3.7 and up. 
You will first need to install the packages in `requirements.txt`.
You will also need to pull the MOCHA repository [here](https://github.com/anthonywchen/mocha) into the root of this repository.

### Evaluation

### Docker

You can use `docker` to build and run the code, like so:

```bash
docker run --rm -it -v (pwd)/results:/results (docker build -q .)
```

## Beaker

We use AI2's [Beaker](https://beaker.org) for running the evaluator. This
requires three things:

1. A Beaker Image, which captures the code for the evaluator.
2. 3 datasets, each of which is a single `.jsonl` file:
    1. The questions
    2. The answers
    3. The predictions


There are instructions for creating each below.

## Creating the Beaker Image

To push a new Beaker image, which captures the evaluator code, run:

```bash
# Build a new version with the latest changes
docker build -t mocha-eval:latest

# Rename the current image named "mocha-eval" so we can push a new version with
# that name.
beaker image rename mocha-eval mocha-eval-$(date -u "+%Y-%m-%dT%H:%M:%SZ")"

# Push the version we just built to Beaker and give it the name "mocha-eval"
beaker image create mocha-eval:latest -n mocha-eval --workspace ai2/mocha
```

## Creating Datasets

The evaluator requires three datasets as input. The commands below show
how to create these datasets from dummy input (that's used for validating
the evaluator):

```bash

beaker dataset rename \
    mocha-dummy-predictions \
    mocha-dummy-predictions-$(date -u "+%Y-%m-%dT%H:%M:%SZ")"
beaker dataset create \
    mocha-dummy-predictions \
    evaluator/test_predictions.jsonl \
    --workspace ai2/mocha

beaker dataset rename \
    mocha-dummy-questions \
    mocha-dummy-questions-$(date -u "+%Y-%m-%dT%H:%M:%SZ")"
beaker dataset create \
    mocha-dummy-questions \
    evaluator/test_questions.jsonl \
    --workspace ai2/mocha

beaker dataset rename \
    mocha-dummy-answers \
    mocha-dummy-answers-$(date -u "+%Y-%m-%dT%H:%M:%SZ")"
beaker dataset create \
    mocha-dummy-answers \
    evaluator/test_answers.jsonl \
    --workspace ai2/mocha
```

## Running an Experiment

After updating the Beaker image or the datasets you can verify that things
are working by running an experiment. Do do so run:

```bash
beaker experiment create beaker.yaml --worksapce ai2/mocha
```

That command will return a URL like [this one](https://beaker.org/ex/01FHZWDM4WP2XDC3AX1Y11ZM76/tasks/01FHZWDM527J6248TBPZM1F0T9),
which you can use to monitor the progress.

Once the experiment is complete download the results and make sure they're
correct.

