import requests

#for testing 
for_prediction_dict = {
    "pclass":3,
    "sex":1,
    "age":13,
    "sibsp":1,
    "parch":2,
    "fare":5
}

url = 'http://127.0.0.1:8000/titanic_sync'

response = requests.post(url, json=for_prediction_dict)
#response.status_code
print(response.json())