import tensorflow as tf
from base_exercise import BaseExercise

class Exercise1(BaseExercise):
    """
    This exercise trains a Deep Neural Network (DNN) on the Fashion-MNIST dataset and evaluates its performance.

    Assignment requirements:
    - 3 hidden layers
    - 100 neurons per layer
    - ELU activation function
    - He weighted initialization
    - Nadam optimizer
    - Early stopping
    """

    def __init__(self):
        """
        Constructor.
        Initialize variables that will later store data, labels, model, and training history.
        """
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.history = None

    def load_data(self):
        """
        Load and prepare Fashion-MNIST data.

        Fashion-MNIST contains grayscale images of clothing.
        Each image is 28x28 pixels and belongs to one of 10 categories.
        """
        print("Loading Fashion-MNIST data...")
        # Download Fashion-MNIST dataset.
        (X_train, y_train), (X_test, y_test) = (tf.keras.datasets.fashion_mnist.load_data())

        # Use only the first 2000 training samples.
        self.X_train = X_train[:2000].astype('float32') / 255.0
        self.y_train = y_train[:2000]
        # Use only the first 400 test samples
        self.X_test = X_test[:400].astype('float32') / 255.0
        self.y_test = y_test[:400]

        print("Data loaded!")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")

    def build_model(self):
        """
        Create the neural network architecture.
        Architecture:
           Flatten
           Dense(100, ELU)
           Dense(100, ELU)
           Dense(100, ELU)
           Dense(10, Softmax)
        """
        print("Building model...")
        self.model = tf.keras.Sequential([
            # Convert the 28x28 image to a one dimensional vector.
            tf.keras.layers.Flatten(input_shape=(28, 28)),

            # Hidden Layer 1
            # ELU helps improve gradient flow
            # He initialization for deep neural networks using ReLU family
            tf.keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),

            # Hidden Layer 2
            tf.keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),

            # Hidden layer 3
            tf.keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),

            # Output layer
            # Fashion MNIST has 10 classes, therefore need 10 output.
            # Softmax converts output into class probabilities.
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        # Configure how the model learns.
        self.model.compile(
            #Nadam combines Adam optimization with Nesterov momentum.
            optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),

            # Suitable when labels are integer encoded.
            loss="sparse_categorical_crossentropy",

            # Track accuracy during training.
            metrics=["accuracy"]
        )
        print("Model created!")

        # Display model architecture.
        self.model.summary()

    def train(self):
        """
        Train the neural network.

        Early stopping prevents unnecessary training and helps reduce overfitting.
        """
        print("Starting training...")

        # EarlyStopping callback when validation loss stops improving.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            # Wait 5 epochs before stopping.
            patience=5,
            # Restore the best-performing weights.
            restore_best_weights=True
        )

        self.history = self.model.fit(
            self.X_train, self.y_train,

            # Use 10% of training data for validation
            validation_split=0.1,
            # Maximum number of epochs.
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        print("Training finished!")

    def evaluate(self):
        """
        Evaluate the neural network on unseen test data.
        """
        print("Starting evaluation...")

        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("\nResults:")
        print("---------------------------")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Q1.1 Test Accuracy: {test_acc*100:.2f}%")

        return test_acc

    def plot_results(self):
        """
        Visualize training and validation history.
        To help identify over- and underfitting.
        """

        self.plot_history(
            self.history, "Exercise 1 - Fashion MNIST Training History"
        )

    def run(self):
        """
        Run the training and evaluation loop.
        """
        print("\n" + "=" * 50)
        print("Exercise 1")
        print("=" * 50)

        # Ensure reproducible results.
        self.set_seed()

        # Step 1: Load data.
        self.load_data()
        # Step 2: Create model.
        self.build_model()
        # Step 3: Train model.
        self.train()
        # Step 4: Evaluate model
        self.evaluate()
        # Step 5: Plot training history.
        self.plot_results()

