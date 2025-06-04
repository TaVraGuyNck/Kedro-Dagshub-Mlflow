FROM public.ecr.aws/lambda/python:3.9

WORKDIR /var/task

COPY requirements.txt .

COPY models/ models/
COPY handler.py .
COPY preprocessor.py .
COPY startupdelay_horizon/ startupdelay_horizon/

RUN mkdir -p /var/tmp

RUN pip install --default-timeout=100 -r requirements.txt

CMD ["handler.lambda_handler"]


