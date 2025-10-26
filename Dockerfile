FROM python:3.9-slim-bookworm
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]