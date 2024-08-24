import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images to [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to add a channel dimension (needed for Conv2D)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Build and compile the CNN model for MNIST
mnist_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

mnist_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Train the MNIST model
history = mnist_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the MNIST model
test_loss, test_acc = mnist_model.evaluate(test_images, test_labels)
print(f"MNIST Test accuracy: {test_acc:.4f}")

# Predict and display MNIST images
predictions = mnist_model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
# Display training and validation accuracy for MNIST
plt.figure(figsize=(12, 5))

# Subplot for accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('MNIST Accuracy')

# Display the first 10 images from MNIST with predictions
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"T: {test_labels[i]}\nP: {predicted_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save and reload MNIST model
mnist_model.save('mnist_digit_recognition_model.keras')
loaded_mnist_model = tf.keras.models.load_model('mnist_digit_recognition_model.keras')

# Load the EMNIST dataset (letters subset)
(ds_train, ds_test), ds_info = tfds.load('emnist/letters', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

# Function to normalize and resize the images
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1)  # Add channel dimension
    return image, label - 1  # Adjust labels to start from 0

# Prepare the training and testing datasets
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(128).prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128).cache().prefetch(tf.data.experimental.AUTOTUNE)

# Build and compile the CNN model
model = models.Sequential([
    Input(shape=(28, 28, 1)),  # Define the input shape explicitly here
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(ds_train, epochs=5, validation_data=ds_test)

# Evaluate the model
test_loss, test_acc = model.evaluate(ds_test)
print(f"Test accuracy: {test_acc:.4f}")

# Get a batch of test images and labels for display
test_images, test_labels = next(iter(ds_test))

# Make predictions on the batch
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Convert numerical labels to characters
label_map = {i: chr(65 + i) for i in range(26)}  # Map 0-25 to A-Z

# Number of images to display
num_images = 10

# Display the first `num_images` images in the batch with their true and predicted labels
plt.figure(figsize=(15, 2 * num_images))  # Adjust the figure size based on number of images

for i in range(num_images):
    plt.subplot(num_images, 10, i + 1)  # Display `num_images` in a single row
    plt.imshow(test_images[i].numpy().squeeze(), cmap='gray')  # Remove the channel dimension
    
    # True and predicted labels
    true_label = test_labels[i].numpy()
    predicted_label = predicted_labels[i]
    
    # Display true and predicted labels
    plt.title(f"T: {label_map[true_label]}\nP: {label_map[predicted_label]}")
    plt.axis('off')

plt.tight_layout()
plt.show()  # Display the figure with true and predicted labels

# Compute confusion matrix for the first 10 images only
conf_matrix = confusion_matrix(test_labels[:num_images], predicted_labels[:num_images], labels=range(num_images))

# Create a ConfusionMatrixDisplay instance
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=[label_map[i] for i in range(num_images)])

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix for 10 Images")
plt.show()  # Display the confusion matrix