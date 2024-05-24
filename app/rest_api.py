from flask import Flask, jsonify, request, abort
from ML_models import Perceptron
import pickle, os
import numpy
from tests import unit_tests

app = Flask(__name__)


def load_model_dump(name):
    filepath = os.path.join(os.path.dirname(__file__), "ML_models_pickles", "{}.dump".format(name))
    with open(filepath, 'rb') as input_file:
        model = pickle.load(input_file)
    return model


def dump_models_list():
    pickle.dump(models_list, open(os.path.join(os.path.dirname(__file__), "ML_models_pickles", "models_list.sav"), 'wb'))


def dump_model(model, name):
    pickle.dump(model, open(os.path.join(os.path.dirname(__file__), "ML_models_pickles", "{}.dump".format(name)), 'wb'))


with open(os.path.join(os.path.dirname(__file__), "ML_models_pickles", "models_list.sav"), 'rb') as input_file:
    models_list = pickle.load(input_file)


@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({"models": models_list})


@app.route('/create', methods=['POST'])
def create_model():
    if not request.json or not 'name' in request.json:
        abort(400)
    elif 'eta' in request.json and not isinstance(request.json["eta"], float):
        abort(400)
    elif 'n_iter' in request.json and not isinstance(request.json["n_iter"], int):
        abort(400)

    eta = 0.01
    n_iter = 10

    if 'eta' in request.json:
        eta = request.json["eta"]

    if 'n_iter' in request.json:
        n_iter = request.json["n_iter"]

    model = Perceptron(eta, n_iter)
    name = request.json["name"]


    dump_model(model, name)

    new_model = {
        "id": models_list[-1]["id"] + 1 if models_list else 1,
        "name": request.json["name"]
    }

    models_list.append(new_model)
    dump_models_list()
    return jsonify(new_model), 201


@app.route('/fit/<int:model_id>', methods=['PUT'])
def model_fit(model_id):
    model_index = next((model_index for model_index in models_list if model_index["id"] == model_id), None)
    if model_index is None:
        abort(404)
    if not request.json or not 'X' in request.json or not 'y' in request.json:
        abort(400)

    model = load_model_dump(model_index["name"])

    X = numpy.array(request.json["X"]).reshape((-1, 1))
    y = numpy.array(request.json["y"])

    model.fit(X, y)

    res = model_index
    res['errors'] = model.errors_

    dump_model(model, model_index["name"])
    return jsonify(res)


@app.route('/predict/<int:model_id>', methods=['PUT'])
def model_predict(model_id):
    model_index = next((model_index for model_index in models_list if model_index["id"] == model_id), None)
    if model_index is None:
        abort(404)
    if not request.json or not 'X' in request.json:
        abort(400)

    model = load_model_dump(model_index["name"])

    X = numpy.array(request.json["X"]).reshape((-1, 1))

    preds = model.predict(X)

    res = model_index
    res['predictions'] = preds.tolist()

    return jsonify(res)


@app.route('/delete/<int:model_id>', methods=['DELETE'])
def model_delete(model_id):
    model_index = next((model_index for model_index in models_list if model_index["id"] == model_id), None)
    if model_index is None:
        abort(404)

    os.remove(os.path.join(os.path.dirname(__file__), "ML_models_pickles", "{}.dump".format(model_index["name"])))
    models_list.remove(model_index)
    dump_models_list()

    return jsonify({"result": True})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
