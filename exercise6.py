import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from base_exercise import BaseExercise

class Exercise6(BaseExercise):
    """
    CNN with Monte Carlo dropout.

    This exercise evaluates:
        Standard classification accuracy.
        MC dropout classification accuracy.
        Epistemic uncertainty.

    Monte Carlo dropout allows ut to estimate how confident the model is in its predictions.
    """

    def __init__(self):
        self.train_ds = None
        self.test_ds = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.history = None

    def load_data(self):
        """
        Load and prepare KMNIST dataset.
        """
        print("Loading KMNIST dataset...")

        train_ds, test_ds = tfds.load(
            "kmnist",
            split=["train[:3000]", "test[:500]"],
            as_supervised=True,
            shuffle_files=False
        )

        def preprocess(image, label):

            image = tf.cast(image, tf.float32) / 255.0

            return image, label

        self.train_ds = (
            train_ds
            .map(preprocess)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.test_ds = (
            test_ds
            .map(preprocess)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Get NumPy arrays to MC dropout analysis.
        X_test_list = []
        y_test_list = []

        for image, label in self.test_ds:

            X_test_list.append(image.numpy())

            y_test_list.append(label.numpy())

        self.X_test = np.concatenate(X_test_list, axis=0)

        self.y_test = np.concatenate(y_test_list, axis=0)

    def build_model(self):
        """
        Build CNN architecture.
        Dropout layers are included because they are needed later for Monte Carlo dropout.
        """
        print("Building CNN model...")

        self.model = tf.keras.Sequential([
            # Feature extraction
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Classification
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation='relu'),

            # Dropout randomly disables neurons.
            # During MC dropout evaluation this layer remains active.
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Model built!")
        self.model.summary()

    def train(self):
        """
        Train the model.
        """
        print("Training model...")

        self.history = self.model.fit(
            self.train_ds,
            epochs=15,
            verbose=1
        )

        print("Training complete!")

    def evaluate_standard(self):
        """
        Standard evaluation.
        During normal evaluation, dropout is automatically disabled.
        """

        print("\nStandard evaluation...")
        print("--------------------------------")

        loss, accuracy = self.model.evaluate(
            self.test_ds,
            verbose=0
        )

        print(f"Q6.1 Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    def mc_dropout_predict(self, n_passes=20):
        """
        The same images are passed through the network multiple times while keeping dropout active.
        This generated multiple slightly different predictions for each sample.
        """

        predictions = np.stack(
            [
                self.model(self.X_test, training=True).numpy()
                for _ in range(n_passes)
            ],
            axis=0
        )
        return predictions

    def evaluate_mc_dropout(self, predictions):
        """
        Evaluate accuracy using MC dropout.
        """
        print("\nEvaluating MC dropout...")
        print("-------------------------------")

        mean_predictions = np.mean(predictions, axis=0)

        predicted_classes = np.argmax(mean_predictions, axis=1)

        mc_accuracy = np.mean(predicted_classes == self.y_test)

        print(f"Q6.2 MC Dropout Accuracy: {mc_accuracy * 100:.2f}%")
        return mc_accuracy

    def calculate_uncertainty(self, predictions):
        """
        Estimate epistemic uncertainty.
        Epistemic uncertainty reflects uncertainty caused by limited model knowledge.
        This variance of predictions is used as an uncertainty estimate.
        """

        print("\nUncertainty analysis")
        print("---------------------------")

        # Calculate variance across all forward passes.
        variance = np.var(predictions, axis=0)

        # Compute uncertainty score for each image.
        sample_uncertainty = np.mean(variance, axis=1)

        # Compute one overall score.
        overall_uncertainty = np.mean(sample_uncertainty)

        print(f"Q6.3 Average Epistemic Uncertainty: {overall_uncertainty:.3f}")
        return overall_uncertainty

    def plot_results(self):
        """
        Visualize training progress.
        """

        self.plot_history(self.history, "Exercise 6 - CNN with MC dropout")

    def run(self):
        """
        Execute the complete workflow.
        """

        print("\n" + "=" * 50)
        print("EXERCISE 6")
        print("=" * 50)

        # Ensure reproducible results.
        self.set_seed()
        # Step 1
        self.load_data()
        # Step 2
        self.build_model()
        # STep 3
        self.train()
        # Step 4
        self.evaluate_standard()
        # Step 5
        predictions = self.mc_dropout_predict(n_passes=20)
        # Step 6
        self.evaluate_mc_dropout(predictions)
        # Step 7
        self.calculate_uncertainty(predictions)
        # Step 8
        self.plot_results()

