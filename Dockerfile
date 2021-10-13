FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "python" ]
CMD [ "evaluator/evaluate.py", \
      "-a", "/app/evaluator/test_answers.jsonl", \
      "-p", "/app/evaluator/test_predictions.jsonl", \
      "-m", "/results/metrics.jsonl" ]

