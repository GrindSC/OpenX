# OpenX

## Solution

Solution for the task is included in solution.py  
It's important to run the script to create models for the api

## Create docker image

To build the Docker image, you can run the following command in the terminal:  
docker build -t rest-api .

## Run container

To run the Docker container, you can use the following command:  
docker run -p 5000:5000 rest-api

## Make a request

To test api sent request to http://localhost:5000/predict or run the request.py
