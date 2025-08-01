# UC Merced Land Use Classifier
## Overview

This repository provides a complete pipeline to classify aerial images from the **UC Merced Land Use Dataset** using transfer learning with a pretrained **MobileNetV2** backbone.

## Motivation

Automated land use classification from remote sensing imagery is crucial for urban planning, environmental monitoring, and agricultural management. Leveraging pretrained models enables high accuracy on domain-specific datasets with fewer samples.

## Features

* **Data Loading & Splitting** via tensorflow_datasets.
* **Data Augmentation**: Random flips, rotations, zooms, and contrast adjustments.
* **Transfer Learning** with MobileNetV2:

  * Custom classification head with dropout and batch normalization.
  * Two-phase training (head training + fine-tuning).
* **Callbacks**: Early stopping and learning rate reduction on plateau.
* **Evaluation**: Classification report, confusion matrix, and final test accuracy.
* **Visualization**: Training & validation loss/accuracy curves.
* **Model Persistence**: Save/Load Keras models (`.h5`).

## Tech Stack & Dependencies

* **Python** ≥ 3.8
* **TensorFlow** ≥ 2.10
* **TensorFlow Datasets**
* **NumPy**
* **scikit-learn**
* **Matplotlib**
* **Seaborn**

## Dataset

The **UC Merced Land Use Dataset** contains 21 classes (100 images each, total 2,100) of 256×256 RGB aerial images.

* **Training split**: 80%
* **Validation/Test split**: 20%

Splits are created on-the-fly using the TFDS slicing API.


## Configuration

Edit the constants at the top of `train.py`:

```
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 21
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 30
LEARNING_RATE_PHASE1 = 1e-3
LEARNING_RATE_PHASE2 = 1e-5
PATIENCE = 10
LR_PATIENCE = 5
```

## Data Preprocessing & Augmentation

* **Resize** to 224×224
* **Normalization**: Cast to `float32` and apply `mobilenet_v2.preprocess_input` (\[–1,1] range)
* **Augmentation** (training only):

  * Random horizontal flip
  * Random rotation (±10%)
  * Random zoom (±10%)
  * Random contrast (±10%)

```
# Example preprocessing for training
def preprocess_train(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = data_augmentation(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label
```

## Model Architecture

```
def create_model(num_classes, fine_tune=False):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = fine_tune

    if fine_tune:
        for layer in base_model.layers[:100]:
            layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

## Training Procedure

### Phase 1: Base Training

1. Freeze the MobileNetV2 backbone (`fine_tune=False`).
2. Compile:

   ```
   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-3),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   ```
3. Callbacks:

   * `EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)`
   * `ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-7)`
4. Train for up to `PHASE1_EPOCHS`:

   ```
   python train.py --phase1
   ```

### Phase 2: Fine-Tuning

1. Unfreeze backbone, freeze first 100 layers.
2. Recompile with lower LR:

   ```
   model.compile(
       optimizer=tf.keras.optimizers.Adam(1e-5),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   ```
3. Continue training for up to `PHASE2_EPOCHS`:

   ```
   python train.py --phase2
   ```

## Evaluation & Metrics

* **Test Accuracy** printed after evaluation on the 20% split.
* **Classification Report** (precision, recall, F1) via `sklearn.metrics.classification_report`.
* **Confusion Matrix** via `sklearn.metrics.confusion_matrix` + optional Seaborn heatmap.


## Plotting Training History

Use the built-in `plot_training_history(history1, history2=None)` to

* Plot accuracy and loss curves
* Annotate fine-tuning start epoch
* Summarize total epochs and best validation accuracy.
  
<img width="1200" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/ae3a0d59-d2a8-479b-82d6-3ac253c8fc87" />

## Project Structure

```
uc_merced_classifier/
├── data/
├── models/
├── train.py         
├── requirements.txt
├── README.md
```

## Acknowledgements

* **UC Merced Land Use Dataset**
* **TensorFlow** & **Keras** teams
* Tutorials on transfer learning with MobileNetV2
* **scikit-learn** for evaluation metrics
