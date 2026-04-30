"""
Custom early stopping callback for BSPNN.
"""

import numpy as np

try:
    import tensorflow.keras as keras
except ImportError:
    import keras


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if current < 1e-4:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Loss is smaller then 1E-06, stopping training.")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            print("val_accuracy:" + str(logs.get("accuracy")))
            if logs.get("accuracy") > 0.99:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Training accuracy is 100%, stopping training.")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    else:
                        print("Warning: No best weights to restore, keeping current weights.")
