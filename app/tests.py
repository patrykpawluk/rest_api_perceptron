import requests


def test_get_models():
    res = requests.get('http://127.0.0.1:5000/models')
    assert(res.status_code == 200)

def test_create_model():
    res = requests.post('http://127.0.0.1:5000/create', json={"name": "test_model", "eta": 0.1, "n_iter": 10})
    assert(res.status_code == 201)

def test_model_fit():
    get_res = requests.get('http://127.0.0.1:5000/models')
    get_res_dict = get_res.json()
    last_id = get_res_dict["models"][len(get_res_dict["models"])-1]["id"]

    X = [1, 2, 3, 4]
    y = [-1, -1, 1, 1]

    res = requests.put("http://127.0.0.1:5000/fit/{}".format(last_id), json={"X": X, "y": y})
    assert(res.status_code == 200)

def test_model_predict():
    get_res = requests.get('http://127.0.0.1:5000/models')
    get_res_dict = get_res.json()
    last_id = get_res_dict["models"][len(get_res_dict["models"])-1]["id"]

    X = [1, 2, 3, 4]
                 
    res = requests.put("http://127.0.0.1:5000/predict/{}".format(last_id), json={"X": X})
    assert(res.status_code == 200)

def test_model_delete():
    get_res = requests.get('http://127.0.0.1:5000/models')
    get_res_dict = get_res.json()
    last_id = get_res_dict["models"][len(get_res_dict["models"])-1]["id"]

    res = requests.delete("http://127.0.0.1:5000/delete/{}".format(last_id))
    assert(res.status_code == 200)

def unit_tests():
    test_get_models()
    print("test_get_models pass")

    test_create_model()
    print("test_create_model pass")

    test_model_fit()
    print("test_model_fit pass")

    test_model_predict()
    print("test_model_predict pass")

    test_model_delete()
    print("test_model_delete pass")
