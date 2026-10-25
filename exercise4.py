import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from datasets import load_dataset
from base_exercise import BaseExercise

class Exercise4(BaseExercise):
    """
    Exercise 4
    Transfer Learning with MobileNetV2 and EuroSAT.
    Instead of training a neural network from scratch, we reuse knowledge learned on the ImageNet dataset.
    MobileNetV2 already knows how to detect useful image features such as edges, textures, shapes, and patterns.
    """
    def __init__(self):
        self.train_ds = None
        self.test_ds = None
        self.model = None
        self.history = None

    def load_data(self):
        """
        Load EuroSAT dataset.
        EuroSAT contains satellite images divided into 10 land use categories such as forests, rivers, highways, and residential areas.
        """
        print("Loading EuroSAT dataset...")

        # Load the dataset split without shuffling to match assignment requirements[cite: 3]
        dataset = load_dataset("blanchon/EuroSAT_RGB", split="train")

        # Select the first 2000 for training and the next 500 for testing[cite: 3]
        train_data = dataset.select(range(2000))
        test_data = dataset.select(range(2000, 2500))

        # Convert to numpy arrays for smooth pipeline creation
        X_train = np.array([np.array(img) for img in train_data["image"]])
        y_train = np.array(train_data["label"])

        X_test = np.array([np.array(img) for img in test_data["image"]])
        y_test = np.array(test_data["label"])

        # Create TensorFlow datasets from tensors
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        print("Data loaded")


       # Had to load the dataset locally, since I was not able to connect to the tfds server.
       # train_ds, test_ds = tfds.load(
       #     "eurosat/rgb",
       #     split=["train[:2000]", "train[2000:2500]"],
       #     as_supervised=True,
       #     shuffle_files=False
       # )

       # self.train_ds = train_ds
       # self.test_ds = test_ds

        #print("Dataset loaded!")

    def preprocess(self, image, label):
        """
        Prepare images for MobileNetV2.
        MobileNetV2 expects: RGB images, 64x64 pixels and floating point values.
        """

        # Resize image to 64x64 pixels.
        image = tf.image.resize(image, (64, 64))

        # Convert values from integers to float. Scale pixel values from [0, 255] to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    def prepare_datasets(self):
        """
        Apply preprocessing and optimize input pipelines.
        """

        # Prepare training data.
        self.train_ds= (
            self.train_ds
            .map(self.preprocess)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Prepare validation data.
        self.test_ds = (
            self.test_ds
            .map(self.preprocess)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

    def build_model(self):
        """
        Create transfer learning model.
        """
        print("Building model...")

        # Load MobileNetV2 without its original classifier.
        # include_top=False removes the ImageNet output layer.
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(64, 64, 3),
            include_top=False,
            weights='imagenet',
        )

        # Freeze pretrained weights.
        # The network will keep the ImageNet knowledge during training.
        base_model.trainable = False

        self.model = tf.keras.Sequential([
            # Pretrained feature extractor.
            base_model,
            # Convert feature maps into a vector.
            tf.keras.layers.GlobalAveragePooling2D(),
            # Small classifier network.
            tf.keras.layers.Dense(128, activation='relu'),

            # Randomly disable neurons during training.
            # This helps reduce overfitting.
            tf.keras.layers.Dropout(0.2),

            # EuroSAT contains 10 classes.
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        print("Model created")
        self.model.summary()

    def train(self):
        """
        Train only the classifier layers.
        The MobileNetV2 feature extractor remains frozen.
        """

        print("Training model...")

        self.history = self.model.fit(
            self.train_ds,
            epochs=5,
            verbose=1
        )

        print("Training complete!")

    def evaluate(self):
        """
        Evaluate the transfer learning model.
        """

        print("Evaluating model...")

        loss, accuracy = self.model.evaluate(
            self.test_ds,
            verbose=0
        )

        print("\nResults")
        print("---------------------")
        print(f"Test loss: {loss:.4f}")
        print(f"Q4.1 Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    def plot_results(self):
        """
        Visualize learning behaviour.
        """

        self.plot_history(self.history, "Exercise 4 - Transfer Learning with MobileNetV2")

    def run(self):
        """
        Execute complete workflow.
        """
        print("\n" + "=" * 50)
        print("EXERCISE 4")
        print("=" * 50)

        # Make experiments reproducible.
        self.set_seed()
        # Step 1
        self.load_data()
        # Step 2
        self.prepare_datasets()
        # Step 3
        self.build_model()
        # Step 4
        self.train()
        # Step 5
        self.evaluate()
        # Step 6
        self.plot_results()