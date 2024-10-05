import requests

url = "https://deploying-a-machine-learning-model-with-to4u.onrender.com"
endpoint = "/predict"

test = {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }

response = requests.post(f"{url}{endpoint}", json=test)
print(f"Prediction for\n\n{test}:\n")
print(f"Predicion: {response.json()['prediction']}\n")
print("·0 - <=50K\n·1 - >50K")
