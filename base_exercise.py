import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class BaseExercise:
    def set_seed(self):
        """
        Set a fixed random seed to make experiments reproducible.

        Using the sam seed ensures that random operations sush as
        weight initialization and data shuffling produce the same
        results every time the code is executed.
        """
        tf.keras.utils.set_random_seed(42)

    def plot_history(self, history, title):
        """
        Plot the training and validation loss over time.
        This graph helps us understand how well the model is learning.
        It can also reveal signs of overfitting or underfitting.
        """

        plt.figure(figsize=(8,5))

        # Plot the loss calculated on hte training dataset.
        # Training loss shows how well the model fits the data it learns from.
        plt.plot(history.history['loss'],
                 label="Training loss")

        # Plot the loss calculated on the validation dataset if it exists.
        # Validation loss indicates how well the model performs on unseen data durin training.
        if "val_loss" in history.history:
            plt.plot(history.history['val_loss'], label="Validation loss")

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        plt.show()