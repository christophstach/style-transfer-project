# stach/stylized-api

FROM python:3.6

WORKDIR /app

COPY ./checkpoints          checkpoints
COPY ./csfnst               csfnst
COPY ./requirements.txt     requirements.txt
COPY ./setup.py             setup.py
COPY ./scripts/server.py    scripts/server.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["scripts/server.py"]