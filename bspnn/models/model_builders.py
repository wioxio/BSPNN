"""
Model architecture builders for BSPNN.
"""

import numpy as np

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except ImportError:
    from keras.models import Sequential
    from keras.layers import Dense


def make_pathway_model(x_shape_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p, pathway_i, pathways_sub_p, diag_self):
    model = Sequential()
    model.add(Dense(x_shape_p, input_dim=x_shape_p, activation="relu", name="model" + str(pathway_i) + "_layer_0", trainable=True))
    model.set_weights([diag_self, np.zeros(len(pathways_sub_p))])
    model.add(Dense(1, activation="relu", name="model" + str(pathway_i) + "_layer_1", trainable=False))
    model.layers[1].set_weights([np.array(pathways_sub_p).reshape(-1, 1), np.zeros(1)])
    model.add(Dense(y_shape_p, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer_p, metrics=["accuracy"])
    return model


def make_original_model(x_shape_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=x_shape_p, activation="relu", name="original_model_layer_0", trainable=True))
    for li in range(1, Nlayers + 1):
        model.add(Dense(Nnodes_p, activation="relu", name="original_model_layer_" + str(li + 1), trainable=True))
    model.add(Dense(y_shape_p, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer_p, metrics=["accuracy"])
    return model


def make_level2_model(npath_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=npath_p, activation="relu", name="level2_model_layer_0", trainable=True))
    for li in range(1, Nlayers + 1):
        model.add(Dense(Nnodes_p, activation="relu", name="level2_model_layer_" + str(li + 1), trainable=True))
    model.add(Dense(y_shape_p, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer_p, metrics=["accuracy"])
    return model
