import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

def read_and_decode(filename, resize_dims):
    # 1. Read the raw file
    img_bytes = tf.io.read_file(filename)
    # 2. Decode image data
    img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
    # 3. Convert pixel values to floats in [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize the image to match desired dimensions
    img = tf.image.resize(img, resize_dims)
    return img

def parse_csvline(csv_line):
    # record_defaults specify the data types for each column
    record_default = ["", ""]
    filename, label_string = tf.io.decode_csv(csv_line, record_default)

    # Load the image
    img = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])

    # Convert label string to integer based on the CLASS_NAMES index
    label = tf.argmax(tf.math.equal(CLASS_NAMES, label_string))
    return img, label

# Define datasets
train_dataset = (
    tf.data.TextLineDataset("gs://cloud-ml-data/img/flower_photos/train_set.csv")
    .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
  )

eval_dataset = (
    tf.data.TextLineDataset("gs://cloud-ml-data/img/flower_photos/eval_set.csv")
    .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

for image_batch, label_batch in train_dataset.take(1): # there are several batches and we can select by entering the batch number
  print("Image batch shape:", image_batch.shape) # num of imgs in a batch, and the dimensions
  print("Label batch shape:", label_batch.shape) # per batch images
  print("Label batch:", label_batch.numpy()) # 0-4, which image are these


for image_batch, label_batch in train_dataset.take(2):
  first_image = image_batch[0].numpy()
  first_label = label_batch[0].numpy()
  print("first image to a label ({})".format(CLASS_NAMES[first_label]))
  plt.imshow(first_image)
  plt.show()

for image_batch, label_batch in train_dataset.take(1):
  fig, axes = plt.subplots(4, 4, figsize=(10, 10)) #4x4 grid

  for i in range(16):
    ax = axes[i // 4, i % 4] # grid position
    ax.imshow(image_batch[i].numpy()) # tensor to numpy
    ax.axis("off")
    ax.set_title(CLASS_NAMES[label_batch[i].numpy()])

  plt.tight_layout()
  plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    keras.layers.Dense(len(CLASS_NAMES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

epoch = 5 # number of times you wanna go through the dataset

history = model.fit(
    train_dataset,
    validation_data = eval_dataset,
    epochs = epoch
)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Take exactly one batch from the evaluation dataset
for images, labels in eval_dataset.take(1):
    # Get model predictions for this batch
    batch_predictions = model.predict(images)
    predicted_indices = np.argmax(batch_predictions, axis=1)

    # Number of images in this batch
    num_images = images.shape[0]

    # Configure how many images to display per row
    num_cols = 4
    num_rows = math.ceil(num_images / num_cols)

    # Create a figure with a suitable size
    plt.figure(figsize=(12, 3 * num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)

        # Display the image
        plt.imshow(images[i].numpy())
        plt.axis('off')

        # Get predicted and actual class names
        pred_class = CLASS_NAMES[predicted_indices[i]]
        actual_class = CLASS_NAMES[labels[i].numpy()]

        # Show both predicted and actual labels as title
        plt.title(f"Pred: {pred_class}\nActual: {actual_class}", fontsize=10)

        if predicted_indices[i] == labels[i].numpy():
          # Create a Rectangle patch
          rect = patches.Rectangle((0, 0), images[i].shape[1]-1, images[i].shape[0]-1,
                                     linewidth=2, edgecolor='g', facecolor='none')

          # Add the rectangle to the subplot axes
          ax.add_patch(rect)

    # Adjust spacing to avoid overlapping titles, etc.
    plt.tight_layout()
    plt.show()

