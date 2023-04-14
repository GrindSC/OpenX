import requests
import pandas as pd

## Testing
# For testing I use the exaples from dataset but it can be replaced with user data
df = pd.read_csv('covtype.data', header=None)
df = df.drop(columns=[54])
df = df.iloc[:10]
data=df.values.tolist()

# Make request to api and display predictions
response = requests.post('http://localhost:5000/predict', json={'model': 'heuristic', 'inputs': data})
print(response.json())

response = requests.post('http://localhost:5000/predict', json={'model': 'baseline1', 'inputs': data})
print(response.json())

response = requests.post('http://localhost:5000/predict', json={'model': 'baseline2', 'inputs': data})
print(response.json())

response = requests.post('http://localhost:5000/predict', json={'model': 'nn', 'inputs': data})
print(response.json())