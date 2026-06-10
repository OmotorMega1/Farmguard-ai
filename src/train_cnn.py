# This script trains a MobileNetV2 model to detect crop diseases
# from leaf images across 6 classes (Tomato, Pepper, Potato)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")

os.chdir("..")

# CONFIGURATION
CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "num_classes": 6,
    "train_dir": "data/split/train",
    "val_dir": "data/split/val",
    "test_dir": "data/split/test",
    "model_save_path": "models/disease_model.h5",
}

# CLASSS NAMES
CLASS_NAMES = [
    "Tomato Early Blight",
    "Tomato Healthy",
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Potato Early Blight",
    "Potato Healthy",
]

CLASSES = {
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_healthy": "Tomato Healthy",
    "Pepper__bell___Bacterial_spot": "Pepper Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___healthy": "Potato Healthy",
}


# SEVERITY MAPPING
# Derives disease severity from model confidence score
def get_severity(confidence):
    if confidence >= 0.90:
        return "Severe"
    elif confidence >= 0.60:
        return "Moderate"
    else:
        return "Mild"


# DATA AUGMENTATION
# creates image variations
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# normalize pixel values
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training images from folder
train_generator = train_datagen.flow_from_directory(
    CONFIG["train_dir"],
    target_size=CONFIG["image_size"],  # resize all images to 224x224
    batch_size=CONFIG["batch_size"],  # load 32 images at a time
    class_mode="categorical",
)

# Load validation images — used to monitor training progress
val_generator = val_test_datagen.flow_from_directory(
    CONFIG["val_dir"],
    target_size=CONFIG["image_size"],
    batch_size=CONFIG["batch_size"],
    class_mode="categorical",
)

# Load test images — used ONCE at the end for final evaluation
test_generator = val_test_datagen.flow_from_directory(
    CONFIG["test_dir"],
    target_size=CONFIG["image_size"],
    batch_size=CONFIG["batch_size"],
    class_mode="categorical",
    shuffle=False,
)

# Load MobileNetV2 with pretrained ImageNet weights, excluding the top classification layer
base_model = MobileNetV2(
    input_shape=(224, 224, 3),  # 224x224 RGB image
    include_top=False,  # remove original 1000-class head
    weights="imagenet",
)

# Freeze all MobileNetV2 layers
base_model.trainable = False

# Add our custom layers on top of frozen MobileNetV2
x = base_model.output

# Flatten 3D feature map into 1D vector
x = GlobalAveragePooling2D()(x)

# Hidden layer — learns our specific disease patterns
# L2 regularization (lambda=0.01) prevents overfitting
x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
output = Dense(
    CONFIG["num_classes"], activation="softmax", kernel_regularizer=l2(0.01)
)(x)

# Connect input to output into one complete model
model = Model(inputs=base_model.input, outputs=output)

# Compile model with Adam optimizer and categorical crossentropy loss
# Adam automatically adjusts learning rate per parameter
# categorical_crossentropy is the standard loss for multiclass problems
model.compile(
    optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
class_weight_dict = dict(enumerate(class_weights))

# CALLBACKS
# Functions that run automatically after each epoch
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=CONFIG["model_save_path"],
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# TRAIN THE MODEL
# Feed training images through the model epoch by epoch
history = model.fit(
    train_generator,
    epochs=CONFIG["epochs"],
    validation_data=val_generator,  # monitor J_cv after each epoch
    class_weight=class_weight_dict,  # apply class weights for imbalance
    callbacks=callbacks,  # early stopping + model checkpoint
    verbose=1,
)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

model.save(CONFIG["model_save_path"])
print(f"Model saved to {CONFIG['model_save_path']}")
