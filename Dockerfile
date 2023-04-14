FROM python:3.11.3

WORKDIR /api

COPY . /api

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "rest_api.py"]