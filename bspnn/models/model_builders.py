"""
Model architecture builders for pathway-based neural network models.
"""

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except ImportError:
    # Fallback for older TensorFlow versions or standalone Keras
    from keras.models import Sequential
    from keras.layers import Dense

import numpy as np


def make_pathway_model(x_shape_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p, pathway_i, pathways_sub_p, diag_self):
    """
    Create a pathway-specific model with fixed weights for pathway structure.
    
    Args:
        x_shape_p: Input dimension (number of genes in pathway)
        y_shape_p: Output dimension (typically 1 for binary classification)
        Nnodes_p: Number of nodes in hidden layers (not used in pathway model but kept for consistency)
        Nlayers: Number of layers (not used in pathway model but kept for consistency)
        optimizer_p: Optimizer name (e.g., 'adam', 'sgd', 'rmsprop')
        pathway_i: Pathway index for naming layers
        pathways_sub_p: Pathway weights (numpy array)
        diag_self: Diagonal matrix for first layer weights
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Dense(x_shape_p, input_dim=x_shape_p, activation='relu', 
                    name=f"model{pathway_i}_layer_0", trainable=True))
    model.set_weights([diag_self, np.zeros(len(pathways_sub_p))])
    
    model.add(Dense(1, activation='relu', name=f"model{pathway_i}_layer_1", trainable=False))
    model.layers[1].set_weights([np.array(pathways_sub_p).reshape(-1, 1), np.zeros(1)])
    
    model.add(Dense(y_shape_p, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer=optimizer_p, metrics=['accuracy'])
    return model


def make_original_model(x_shape_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    """
    Create a standard fully-connected neural network model.
    
    Args:
        x_shape_p: Input dimension (number of features/genes)
        y_shape_p: Output dimension (typically 1 for binary classification)
        Nnodes_p: Number of nodes in each hidden layer
        Nlayers: Number of hidden layers
        optimizer_p: Optimizer name (e.g., 'adam', 'sgd', 'rmsprop')
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=x_shape_p, activation='relu', 
                    name="original_model_layer_0", trainable=True))
    
    for li in range(1, Nlayers + 1):
        model.add(Dense(Nnodes_p, activation='relu', 
                        name=f"original_model_layer_{li+1}", trainable=True))
    
    model.add(Dense(y_shape_p, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer=optimizer_p, metrics=['accuracy'])
    return model


def make_level2_model(npath_p, y_shape_p, Nnodes_p, Nlayers, optimizer_p):
    """
    Create a level 2 model that takes pathway predictions as input.
    
    Args:
        npath_p: Number of pathways (input dimension)
        y_shape_p: Output dimension (typically 1 for binary classification)
        Nnodes_p: Number of nodes in each hidden layer
        Nlayers: Number of hidden layers
        optimizer_p: Optimizer name (e.g., 'adam', 'sgd', 'rmsprop')
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Dense(Nnodes_p, input_dim=npath_p, activation='relu', 
                    name="level2_model_layer_0", trainable=True))
    
    for li in range(1, Nlayers + 1):
        model.add(Dense(Nnodes_p, activation='relu', 
                        name=f"level2_model_layer_{li+1}", trainable=True))
    
    model.add(Dense(y_shape_p, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer=optimizer_p, metrics=['accuracy'])
    return model
