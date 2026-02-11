"""
Custom early stopping callback that stops training when loss reaches minimum.
"""

try:
    import tensorflow.keras as keras
except ImportError:
    # Fallback for older TensorFlow versions or standalone Keras
    import keras

import numpy as np


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """
    Stop training when the loss is at its minimum, i.e. the loss stops decreasing.
    
    This callback monitors the loss and stops training when the loss stops decreasing.
    It's similar to EarlyStopping but specifically designed to stop at minimum loss.
    """
    
    def __init__(self, patience=0, verbose=0):
        """
        Args:
            patience: Number of epochs to wait after min loss has been hit.
                     After this number of no improvement, training stops.
            verbose: Verbosity mode, 0 or 1.
        """
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.best_weights = None
        self.best_loss = np.Inf
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        """Initialize variables at the start of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        current_loss = logs.get('loss')
        if current_loss is None:
            return

        if np.less(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.wait = 0
            # Record the best weights
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: early stopping')
                # Restore best weights
                if self.best_weights is not None:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Training stopped at epoch {self.stopped_epoch + 1}')
