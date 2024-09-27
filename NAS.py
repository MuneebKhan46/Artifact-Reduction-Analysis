import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from keras_tuner import HyperModel, Hyperband


def extract_y_channel_from_yuv_with_patch_numbers(yuv_file_path: str, width: int, height: int):
    y_size = width * height
    patches = []

    if not os.path.exists(yuv_file_path):
        print(f"Warning: File {yuv_file_path} does not exist.")
        return [], []

    with open(yuv_file_path, 'rb') as f:
        y_data = f.read(y_size)

    if len(y_data) != y_size:
        print(f"Warning: Expected {y_size} bytes, got {len(y_data)} bytes.")
        return [], []

    y_channel = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))

    for i in range(0, height, 224):
        for j in range(0, width, 224):
            patch = y_channel[i:i+224, j:j+224]
            if patch.shape[0] < 224 or patch.shape[1] < 224:
                patch = np.pad(patch, ((0, 224 - patch.shape[0]), (0, 224 - patch.shape[1])), 'constant')
            patches.append(patch)
    return patches


def load_data_from_csv(csv_path, denoised_dir):
    df = pd.read_csv(csv_path)
    
    all_denoised_patches = []
    all_scores = []
 
    for _, row in df.iterrows():
        denoised_file_name = f"denoised_{row['image_name']}.raw"
        denoised_path = os.path.join(denoised_dir, denoised_file_name)
        
        denoised_patches = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])
        all_denoised_patches.extend(denoised_patches)
      
        patch_scores = row['patch_score'].strip('[]').split(', ')
        scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])
      
        if len(scores) != len(denoised_patches):
          print(f"Error: Mismatch in number of patches and scores for {row['image_name']}")
          continue
        all_scores.extend(scores)

    return all_denoised_patches, all_scores

def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    data = np.expand_dims(data, axis=-1)
    lbl = np.array(labels)
    return data, lbl


denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
csv_path = '/ghosting-artifact-metric/Code/WACV/HighFreq/high_frequency_classification_label.csv'

denoised_patches, labels = load_data_from_csv(csv_path, denoised_dir)
denoised_patches_np, labels_np = prepare_data(denoised_patches, labels)

X_minority = denoised_patches_np[labels_np == 1]
y_minority = labels_np[labels_np == 1]
X_majority = denoised_patches_np[labels_np == 0]
y_majority = labels_np[labels_np == 0]


num_samples_to_generate = len(y_majority) - len(y_minority)


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


augmented_images = []
augmented_labels = []


for _ in range(num_samples_to_generate // len(X_minority) + 1):
    for x in X_minority:
        x = x.reshape((1, 224, 224, 1))
        for x_augmented in datagen.flow(x, batch_size=1):
            augmented_images.append(x_augmented[0])
            augmented_labels.append(1)
            if len(augmented_labels) >= num_samples_to_generate:
                break
        if len(augmented_labels) >= num_samples_to_generate:
            break
    if len(augmented_labels) >= num_samples_to_generate:
        break


X_balanced = np.concatenate((X_majority, X_minority, np.array(augmented_images)))
y_balanced = np.concatenate((y_majority, y_minority, np.array(augmented_labels)))


X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




