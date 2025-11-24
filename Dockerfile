FROM python:3.7-slim

RUN apt-get update && apt-get install -y libsm6 libglib2.0-0 libxrender1 libxext6

COPY docker-requirements.txt .
RUN python3 -m pip install -r docker-requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT ["python3", "/app/easy-walk.py"]
# Default arguments for running on the CORE I/O
CMD [ "192.168.80.101"]
