import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'uc_merced',
    split=['train[:80%]', 'train[80%:]'],
    with_info = True,
    as_supervised = True,
    shuffle_files = True
)
print(f"Number of classes: {ds_info.features['label'].num_classes}")
print(f"Training samples: {ds_info.splits['train'].num_examples * 0.8:.0f}")
print(f"Test samples: {ds_info.splits['train'].num_examples * 0.2:.0f}")

# Configuration
IMG_SIZE = (224, 224)  # Standard size for MobileNetV2
BATCH_SIZE = 32
NUM_CLASSES = ds_info.features['label'].num_classes

# Data augmentation (only for training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# Preprocessing functions
def preprocess_train(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    # Apply augmentation
    image = data_augmentation(image)
    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def preprocess_val(image, label):
    # Resize and normalize
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# Apply preprocessing
ds_train_processed = ds_train.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
ds_train_processed = ds_train_processed.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test_processed = ds_test.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
ds_test_processed = ds_test_processed.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build improved model
def create_model(num_classes, fine_tune=False):
    # Load pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        weights = 'imagenet',
        include_top = False,
        input_shape = IMG_SIZE + (3,)
    )
    
    # Initially freeze the base model
    base_model.trainable = fine_tune
    
    # If fine-tuning, only train the top layers
    if fine_tune:
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    # Build the model
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

# Phase 1: Train with frozen base model
print("Phase 1: Training with frozen base model...")
model = create_model(NUM_CLASSES, fine_tune=False)
model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    patience = 10,
    restore_best_weights = True,
    verbose = 1
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor ='val_loss',
    patience = 5,
    factor = 0.5,
    verbose = 1,
    min_lr = 1e-7
)

# Train phase 1
history1 = model.fit(
    ds_train_processed,
    epochs = 20,
    validation_data = ds_test_processed,
    callbacks = [early_stopping, lr_scheduler],
    verbose = 1
)

# Phase 2: Fine-tuning
print("\nPhase 2: Fine-tuning...")
# Unfreeze the top layers of the base model
base_model = model.layers[0]
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history2 = model.fit(
    ds_train_processed,
    epochs = 30,
    validation_data = ds_test_processed,
    callbacks = [early_stopping, lr_scheduler],
    verbose = 1,
    initial_epoch = len(history1.history['loss'])
)

# Evaluate final model
print("\nFinal Evaluation:")
loss, acc = model.evaluate(ds_test_processed, verbose=1)
print(f"Test Accuracy: {acc * 100:.2f}%")

model.save('uc_merced_classifier.h5')
print("\nModel saved as 'uc_merced_classifier.h5'")

# Fixed plotting function that handles when Phase 2 doesn't run

def plot_training_history(history1, history2=None):
    plt.figure(figsize=(12, 4))
    
    # Check if Phase 2 actually ran any epochs
    if history2 and len(history2.history.get('accuracy', [])) > 0:
        # Combine histories if fine-tuning actually happened
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
        
        # Mark where fine-tuning started
        fine_tune_epoch = len(history1.history['loss'])
        title_suffix = " (with Fine-tuning)"
    else:
        # Only Phase 1 data
        acc = history1.history['accuracy']
        val_acc = history1.history['val_accuracy']
        loss = history1.history['loss']
        val_loss = history1.history['val_loss']
        fine_tune_epoch = None
        title_suffix = " (Phase 1 Only - Phase 2 stopped early)"
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if fine_tune_epoch:
        plt.axvline(x=fine_tune_epoch, color='r', linestyle='--', label='Fine-tuning starts')
    plt.title('Model Accuracy' + title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    if fine_tune_epoch:
        plt.axvline(x=fine_tune_epoch, color='r', linestyle='--', label='Fine-tuning starts')
    plt.title('Model Loss' + title_suffix)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"Phase 1 epochs: {len(history1.history['loss'])}")
    if history2 and len(history2.history.get('accuracy', [])) > 0:
        print(f"Phase 2 epochs: {len(history2.history['loss'])}")
        print(f"Total epochs: {len(acc)}")
    else:
        print(f"Phase 2 epochs: 0 (early stopping triggered)")
        print(f"Total epochs: {len(acc)}")
    
    print(f"Best validation accuracy: {max(val_acc):.4f}")

# Call the fixed plotting function
plot_training_history(history1, history2)

# Save the final model
"""model.save('improved_uc_merced_classifier.keras')
print("Model saved as 'improved_uc_merced_classifier.keras'")"""

# 15. Classification report on test set
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get predictions efficiently in batches
y_true = []
y_pred = []

for batch_imgs, batch_labels in ds_test_processed:
    predictions = model.predict(batch_imgs, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(batch_labels.numpy())

# Print classification report
print(classification_report(y_true, y_pred, target_names=ds_info.features['label'].names))