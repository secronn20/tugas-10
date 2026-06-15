import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    print("Setting up evaluation...")
    
    # Create directories if they do not exist
    os.makedirs('static/assets', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    img_size = (160, 160)
    test_dir = 'dataset/test'
    model_path = 'model/butterfly_model.keras'
    classes_path = 'model/classes.txt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_model.py first.")
        return

    # Load class names
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Fallback to loading classes from directories
        class_names = sorted(os.listdir(test_dir))
    print(f"Target classes: {class_names}")

    # Load test dataset as a batch so we can evaluate all at once
    print("Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=32, # small enough to fit all 25 images in one batch
        label_mode='categorical',
        shuffle=False
    )

    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Extract all images and labels from the test dataset
    test_images = []
    test_labels = []
    for imgs, lbls in test_ds:
        test_images.append(imgs.numpy())
        test_labels.append(lbls.numpy())
        
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Get true class indices
    y_true = np.argmax(test_labels, axis=1)

    # Run predictions
    print("Running predictions on test set...")
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)

    # Compute overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save metrics to JSON file for Flask integration
    metrics = {
        'accuracy': float(accuracy),
        'classes': class_names,
        'report': report,
        'confusion_matrix': cm.tolist()
    }
    
    metrics_path = 'model/evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics to {metrics_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    
    # Translate class names for plotting
    class_names_id = [
        c.replace('ZEBRA_LONG_WING', 'Sayap Panjang Zebra')
         .replace('ADONIS', 'Biru Adonis')
         .replace('MESTRA', 'Pale Mestra')
         .replace('MONARCH', 'Raja (Monarch)')
         .replace('PEACOCK', 'Merak (Peacock)')
        for c in class_names
    ]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names_id, yticklabels=class_names_id,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title('Matriks Kebingungan - Dataset Uji', fontsize=14, pad=15)
    plt.ylabel('Kategori Sebenarnya', fontsize=12)
    plt.xlabel('Kategori Prediksi', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_plot_path = 'static/assets/confusion_matrix.png'
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")

if __name__ == '__main__':
    evaluate()
