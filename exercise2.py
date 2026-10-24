import tensorflow as tf

from base_exercise import BaseExercise

class Exercise2(BaseExercise):
    """
    Exercise 2

    This exercise trains a Deep Neural Network (DNN) ont he CIFAR-100 datat set using coarse labels.
    Coarse labels group objects into 20 board categories, which makes the classification task easier than using all 100 fine-grained classes.
    """

    def __init__(self):
        # Training data
        self.X_train = None
        self.y_train = None
        # Test data
        self.X_test = None
        self.y_test = None
        # Neural network model
        self.model = None
        # Training history
        self.history = None

    def load_data(self):
        """
        Load and prepare the CIFAR-100 dataset.
        The dataset contains color images of size 32x32 pixels. Each image belongs to one of 20 coarse categories.
        """
        print("Loading CIFAR-100 dataset...")

        # Download CIFAR-100 using coarse labels.
        # Examples of coarse labels are: mammals, vehicles, flowers, etc.
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")

        # Store data in class variables.
        self.X_train = X_train[:5000].astype("float32") / 255.0
        self.y_train = y_train[:5000]
        self.X_test = X_test[:1000].astype("float32") / 255.0
        self.y_test = y_test[:1000]

        print("Data loaded!")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")

    def build_model(self):
        """
        Build the Deep Neural Network (DNN) model.
        """
        print("Building Deep Neural Network...")
        self.model = tf.keras.models.Sequential([
            # Convert each 32x32x3 image into a one dimensional feature vector.
            # A 32x32x3 = 3072 values.
            tf.keras.layers.Flatten(input_shape=(32, 32, 3)),

            # Hidden Layer 1
            tf.keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),

            # Hidden Layer 2
            tf.keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),

            # Hidden Layer 3
            tf.keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),

            # Hidden Layer 4
            tf.keras.layers.Dense(256, activation="elu", kernel_initializer="he_normal"),

            # Output Layer
            # Softmax converts outputs into probabilities,
            tf.keras.layers.Dense(20, activation="softmax")
        ])

        # Configure the learning process.
        self.model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                           # Integer labels: 0,1,2,...,19
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"]
                           )
        print("Model created!")
        self.model.summary()

    def train(self):
        """
        Train the Deep Neural Network (DNN) model.
        """
        print("Training Deep Neural Network...")

        # Stop training if validation loss stops improving.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            # Wait 5 epochs before stopping.
            patience=5,
            # Restore the best performing model.
            restore_best_weights=True
            )

        self.history = self.model.fit(

            self.X_train,
            self.y_train,
            # Use 10% of training data for validation.
            validation_split=0.1,
            epochs=50,
            batch_size=128,
            callbacks=[early_stopping],
            verbose=1
        )
        print("Training completed!")

    def evaluate(self):
        """
        Evaluate the Deep Neural Network (DNN) model using unseen test data.
        """
        print("Evaluating Deep Neural Network...")

        test_loss, test_acc = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=0
        )
        print("\nResults")
        print("-------------------------")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Q2.1 Test accuracy: {test_acc*100:.2f}%")

        return test_acc

    def plot_results(self):
        """
        Plot training and validation loss.
        Helps to identify over- and underfitting
        """
        self.plot_history(
            self.history, "Exercise 2 - CIFAR-100 Training History"
        )

    def run(self):
        """
        Execute the exercise 2
        """
        print("\n" + "=" * 50)
        print("EXERCISE 2")
        print("=" * 50)

        # Make results reproducible.
        self.set_seed()
        # Step 1
        self.load_data()
        # Step 2
        self.build_model()
        # Strep 3
        self.train()
        # Strep 4
        self.evaluate()
        # Step 5
        self.plot_results()