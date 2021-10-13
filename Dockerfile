FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# We also need the code that's here: https://github.com/anthonywchen/mocha
# In the nearish future we'll be able to install it as a regular Python
# package, at which point we can remove this hack.
WORKDIR /mocha
RUN wget https://github.com/anthonywchen/MOCHA/archive/refs/heads/main.zip && \
    unzip main.zip && \
    mv MOCHA-main/* . && \
    rm -rf MOCHA-main && \
    rm main.zip && \
    pip install -r requirements.txt

ENV PYTHONPATH /mocha

WORKDIR /app
COPY . .

ENTRYPOINT [ "python" ]
CMD [ "evaluator/evaluate.py", \
      "-a", "/app/evaluator/test_answers.jsonl", \
      "-p", "/app/evaluator/test_predictions.jsonl", \
      "-q", "/app/evaluator/test_questions.jsonl", \
      "-m", "/results/metrics.jsonl" ]

