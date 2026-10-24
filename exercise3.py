import tensorflow as tf
import numpy as np
from base_exercise import BaseExercise

class Exercise3(BaseExercise):
    """
    Exercise 3
    This exercise extends Exercise 1 by creating a self normalizing neural network (SNN).
    Main changes:
    Standardization instead of normalization
    SELU activation
    LeCun initialization
    AlphaDropout
    MC Dropout evaluation
    """
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None

    def load_data(self):
        """
        Load Fashion_MNIST and standardize the data.
        Standardization transforms the data so that it has approximately:
            mean = 0
            std = 1
        """
        print("Loading Fashion_MNIST")

        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        self.X_train = X_train[:2000].astype("float32")
        self.y_train = y_train[:2000]

        self.X_test = X_test[:400].astype("float32")
        self.y_test = y_test[:400]

        # Calculate mean and standard deviation using only the training data.
        # Prevents information leakage form the test set into the training process.
        mean = self.X_train.mean()
        # Small constant added to avoid division by zero.
        std = self.X_train.std() + 1e-7

        # Standardize training data.
        self.X_train = (self.X_train - mean) / std
        # Apply the same transformation to test data.
        self.X_test = (self.X_test - mean) / std

        print("Dataset loaded and standardized")

    def build_model(self):
        """
        Build a self normalizing neural network.
        SELU + LeCun initialization help maintain stable activation during training.
        """
        print("Building model...")

        self.model = tf.keras.Sequential([
            # Convert 28x28 image into a vector.
            tf.keras.layers.Flatten(input_shape=(28, 28)),

            # Hidden Layer 1
            tf.keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            # AlphaDropout perserves the self normalizing property of SELU.
            tf.keras.layers.AlphaDropout(rate = 0.1),

            # Hidden Layer 2
            tf.keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            tf.keras.layers.AlphaDropout(rate = 0.1),

            # Hidden Layer 3
            tf.keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
            tf.keras.layers.AlphaDropout(rate = 0.1),

            # Output layer.
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("Model created")
        self.model.summary()

    def train(self):
        """
        Train the model using early stopping.
        """
        print("Training model...")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        print("Training completed")

    def evaluate(self):
        """
        Evaluate the model using normal inference.
        During normal inference, dropout layers are automatically disabled.
        """

        loss, accuracy = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=0
        )

        print("\nStandard Evaluation")
        print("-----------------------------")
        print(f"Test Loss: {loss:.4f}")
        print(f"Q3.1 Test Accuracy: {accuracy*100:.2f}%")

        return accuracy

    def mc_dropout_predict(self, X, n_passes=20):
        """
        Perform Monte Carlo dropout.
        Instead of disabling dropout during interface, we deliberately keep it active.
        This produces slightly different predictions every time and allows uncertainty estimation.
        """

        predictions = []

        for i in range(n_passes):
            # training=True forces AlphaDropout to remain active.
            prediction = self.model(X, training=True)

            predictions.append(prediction.numpy())

        predictions = np.array(predictions)

        # Calculate the average prediction across all forward passes.
        mean_predictions = np.mean(predictions, axis=0)

        return mean_predictions, predictions

    def evaluate_mc_dropout(self):
        """
        Evaluate the model using MC Dropout.
        """

        mean_predictions, all_predictions = self.mc_dropout_predict( self.X_test, n_passes=20)

        predicted_classes = np.argmax(mean_predictions, axis=1)

        accuracy = np.mean(predicted_classes == self.y_test)

        print("\nMC Dropout Evaluation")
        print("---------------------------")
        print(f"Q3.2 MC Dropout Accuracy: {accuracy * 100:.2f}%")

        return all_predictions

    def calculate_uncertainty(self, predictions):
        """
        Estimate epistemic uncertainty.
        High variance means the modal produces different predictions each time.
        This indicates lower confidence.
        """

        # Calculate variance over all MC forward passes.
        variance = np.var(predictions, axis=0)

        # Average variance across classes.
        sample_uncertainty = np.mean(variance, axis=1)

        # Overall uncertainty score.
        overall_uncertainty = np.mean(sample_uncertainty)

        print("\nUncertainty Analysis")
        print(
            f"Overall Uncertainty: "
            f"{overall_uncertainty:.6f}"
        )

        return overall_uncertainty

    def plot_results(self):
        """
        Plot training and validation loss.
        :return:
        """

        self.plot_history(self.history, "Exercise 3 - Self normalizing network")

    def run(self):

        print("\n" + "=" * 50)
        print("EXERCISE 3")
        print("=" * 50)

        self.set_seed()
        # Step 1
        self.load_data()
        # Step 2
        self.build_model()
        # Step 3
        self.train()
        # Step 4
        self.evaluate()
        # Step 5
        predictions = self.evaluate_mc_dropout()
        # Strep 6
        self.calculate_uncertainty(predictions)
        # Step 7
        self.plot_results()


