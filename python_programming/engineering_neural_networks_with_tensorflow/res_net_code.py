from tensorflow.keras.losses import MSE as mean_squared_error
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import code
import time

def get_resnet_model():
    def residual_block(X, kernels, stride):
        out = tf.keras.layers.Conv1D(kernels, stride, padding='same')(X)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.add([X, out])
        out = tf.keras.layers.ReLU()(out)
        return out

    kernels = 4
    stride = 1

    inputs = tf.keras.layers.Input(shape=(8,1))
    X = tf.keras.layers.Conv1D(kernels, stride)(inputs)
    X = tf.keras.layers.BatchNormalization()(X)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)
    X = residual_block(X, kernels, stride)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    output = tf.keras.layers.Dense(1, activation='linear')(X)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=optimizer,
        loss=mean_squared_error,
        metrics=[mean_squared_error]
    )

    return model

model = get_resnet_model()
X, y = make_regression(n_samples=1000, n_features=8)
X_train, X_test, y_train, y_test = train_test_split(X, y)
start = time.time()
model.fit(X_train, y_train, batch_size=16, epochs=100000)
with open("time_test.txt", "w") as f:
    f.write(f"{time.time() - start}")

