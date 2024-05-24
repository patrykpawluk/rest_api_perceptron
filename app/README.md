Opis projektu

Projekt serwuje aplikację napisaną w Flask, która wystawia rest api pozwalające na tworzenie, trenowanie, liczenie predykcji na prostych sieciach neuronowych(perceptronach)

Lista end pointów api:

endpoint: /models
method: GET
response: {"models": list}

endpoint: /create
method: POST
payload: {"name": str, "eta": float, "n_iter": int}
response: {"id": int, "name": str}

endpoint: /fit/<int:model_id>
method: PUT
payload: {"X": list, "y": list}
response: {"id": int, "name": str, 'errors': list}

endpoint: /predict/<int:model_id>
method: PUT
payload: {"X": list}
response: {"id": int, "name": str, 'predictions': list}

endpoint: delete/<int:model_id>
method: DELETE
response: {"result": bool}

Testy:
Funkcje do testowania kaźdego z endpointów znajdują się w tests.py

Z zewnątrz dokera aplikacja jest dostępna na porcie 5001(http://127.0.0.1:5001/)
