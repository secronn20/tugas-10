import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def train():
    print("Setting up training...")
    
    # Create directories if they do not exist
    os.makedirs('static/assets', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # Load dataset - using smaller size and batch to manage RAM
    img_size = (160, 160)
    batch_size = 8
    
    train_dir = 'dataset/train'
    valid_dir = 'dataset/valid'
    
    print("Loading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    
    print("Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")
    
    # Save class names to a text file for Flask deployment
    with open('model/classes.txt', 'w') as f:
        for c in class_names:
            f.write(c + '\n')
    print("Saved class names to model/classes.txt")

    # Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ])

    # Load MobileNetV2 base model with pre-trained weights, frozen base layers
    print("Loading MobileNetV2 base model...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze pre-trained weights

    # Build model architecture
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./127.5, offset=-1.0)(x)  # MobileNetV2 expects [-1, 1] normalization
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    print("Model summary:")
    model.summary()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save model
    model_path = 'model/butterfly_model.keras'
    model.save(model_path)
    print(f"Model successfully trained and saved to {model_path}")

    # Plot performance curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Akurasi Pelatihan', color='#10b981', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Akurasi Validasi', color='#3b82f6', linewidth=2)
    plt.title('Akurasi Pelatihan dan Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Loss Pelatihan', color='#ef4444', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Loss Validasi', color='#f59e0b', linewidth=2)
    plt.title('Loss Pelatihan dan Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = 'static/assets/training_performance.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Training performance curves saved to {plot_path}")

if __name__ == '__main__':
    train()
