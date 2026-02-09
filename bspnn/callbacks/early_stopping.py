"""
Custom early stopping callback that stops training when loss reaches minimum.
"""

import keras
import numpy as np


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """
    Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """
    
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
    
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        # Initialize best_weights with current weights to avoid NoneType error
        self.best_weights = self.model.get_weights()
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        
        # Stop if loss is very small
        if current < 1E-04:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Loss is smaller than 1E-04, stopping training.")
            return
        
        # Check if loss is decreasing
        if np.less(current, self.best):  # loss is decreasing
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            print("val_accuracy:" + str(logs.get('accuracy')))
            
            # Stop if training accuracy is very high
            if logs.get('accuracy') > 0.99:  # loss didn't change and training accuracy is 1
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Training accuracy is 100%, stopping training.")
            else:  # loss didn't change and training accuracy is less than 1 -> wait until patience
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    else:
                        print("Warning: No best weights to restore, keeping current weights.")
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Training stopped at epoch {self.stopped_epoch + 1}")
