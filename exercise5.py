import tensorflow as tf
import tensorflow_datasets as tfds
from base_exercise import BaseExercise

class Exercise5(BaseExercise):
    """
    Exercise 5
    Convolutional Neural Network (CNN) for image classification.

    Unlike the previous exercises that used fully connected layers,
    this model uses convolutional layers that are specifically designed for image data.

    CNNs preserve spatial relationships between pixels and are therefore much better suited for image classification tasks.
    """

    def __init__(self):
        self.train_ds = None
        self.test_ds = None

        self.model = None
        self.history = None

    def load_data(self):
        """
        Load and prepare KMNIST.
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

    def build_model(self):
        """
        Build the CNN model.
        """
        print("Building CNN model...")

        self.model = tf.keras.Sequential([
            # Convolution Block 1
            # Learn 32 feature detectors.
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

            # Learn more advanced combinations of previously discovered features.
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

            # Reduce image dimensions while preserving the most important features.
            # This decreases computational cost.
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Convolution Block 2
            # Learn more complex patterns.
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Classification Section
            # Convert feature map into a vector that can be processed by Dense layers.
            tf.keras.layers.Flatten(),

            # Fully connected layer.
            # Combines extracted image features into a final representation.
            tf.keras.layers.Dense(256, activation='relu'),

            # Dropout randomly disables neurons during training.
            # Prevents overfitting.
            tf.keras.layers.Dropout(0.3),

            # KMNIST contains 10 character classes.
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

        # Configure training settings.
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("CNN created successfully!")

        self.model.summary()

    def train(self):
        """
        Train the CNN.
        """
        print("Training CNN...")

        self.history = self.model.fit(
            self.train_ds,
            epochs=15,
            verbose=1
        )

        print("Training complete!")

    def evaluate(self):
        """
        Evaluate the model on unseen data.
        """
        print("Evaluating CNN...")

        loss, accuracy = self.model.evaluate(
            self.test_ds,
            verbose=0
        )

        print(f"Q5.1 Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    def plot_results(self):
        """
        Plot learning curves.
        """

        self.plot_history(self.history, "Exercise 5 - CNN KMNIST")

    def run(self):
        """
        Execute the complete workflow.
        """

        print("\n" + "=" * 50)
        print("EXERCISE 5")
        print("=" * 50)

        # Ensure reproducible results.
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
        self.plot_results()