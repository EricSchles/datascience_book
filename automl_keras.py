from sklearn import ensemble
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from sklearn import model_selection
import random
from functools import partial
from collections import defaultdict
import itertools

def baseline_model(neurons, layers):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=10))
    for _ in range(layers):
        model.add(Dense(neurons))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def automl_basic(X_train, X_test, y_train, y_test, baseline, min_neurons, max_neurons, max_layers, num_runs = 3):
    accuracy_scores = defaultdict(list)
    for layers_neurons in itertools.product(range(max_layers), range(min_neurons, max_neurons)):
        layers = layers_neurons[0]
        neurons = layers_neurons[1]
        print("Number of hidden layers", layers)
        for i in range(num_runs):
            deep_broad_model = partial(baseline, neurons, layers)
            estimator = KerasClassifier(build_fn=deep_broad_model, epochs=100, batch_size=5, verbose=0)
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            accuracy_scores[layers_neurons].append(metrics.accuracy_score(y_test, y_pred))
    return accuracy_scores

def choose_best_model(accuracy_scores, min_neurons, max_neurons, max_layers):
    best_acc = 0
    best_layers = 0
    best_neurons = 0
    for layers_neurons in itertools.product(range(max_layers), range(min_neurons, max_neurons)):
        cur_acc = np.mean(accuracy_scores[layers_neurons])
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_layers = layers_neurons[0]
            best_neurons = layers_neurons[1]
    return best_acc, best_layers, best_neurons

random.seed(1)

df = pd.read_csv("housepricedata.csv")
X = df.values[:,0:10]
y = df.values[:,10]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
min_neurons = 8
max_neurons = 13
max_layers = 10
accuracy_scores = automl_basic(X_train, X_test, y_train, y_test, baseline_model, min_neurons, max_neurons, max_layers, num_runs=2)
best_acc, best_layers, best_neurons = choose_best_model(accuracy_scores, min_neurons, max_neurons, max_layers)
print("Optimal number of hidden layers", best_layers)
print("Optimal number of neurons per layer", best_neurons)
print("Optimal accuracy", best_acc)
