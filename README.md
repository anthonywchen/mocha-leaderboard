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

