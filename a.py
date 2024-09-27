
import os
import json
import simplejson
from jsonfix import JsonFix

# Path to your tuner0.json file
tuner_json_path = '/ghosting-artifact-metric/Artifact-Reduction-Analysis/Byesian_Directory/ghosting_artifact_detection/tuner0.json'

# Backup the original file
backup_path = tuner_json_path + '.backup'
if not os.path.exists(backup_path):
    os.rename(tuner_json_path, backup_path)
    print(f"Backup of tuner0.json created at {backup_path}")
else:
    print(f"Backup already exists at {backup_path}")

# Read the corrupted JSON content
with open(backup_path, 'r') as f:
    corrupted_json_content = f.read()

# Attempt to fix the JSON content
try:
    # Try parsing the JSON content to identify errors
    simplejson.loads(corrupted_json_content)
except simplejson.errors.JSONDecodeError as e:
    print(f"JSONDecodeError: {e}")
    print("Attempting to fix the JSON content...")

    fixer = JsonFix()
    fixed_json_content = fixer.fix(corrupted_json_content)

    # Verify if the fixed content is valid JSON
    try:
        json_data = json.loads(fixed_json_content)
        print("JSON content fixed successfully.")
    except json.JSONDecodeError as e:
        print(f"Failed to fix JSON content: {e}")
        print("Automatic repair was unsuccessful.")
        exit(1)

    # Save the fixed JSON content back to tuner0.json
    with open(tuner_json_path, 'w') as f:
        f.write(fixed_json_content)
        print(f"Fixed JSON content saved to {tuner_json_path}")
else:
    print("No JSON errors found in the original file.")



# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import keras_tuner
# from keras_tuner import HyperModel, BayesianOptimization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle

# # Initialize the strategy
# strategy = tf.distribute.MirroredStrategy()
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# def extract_y_channel_from_yuv_with_patch_numbers(yuv_file_path: str, width: int, height: int):
#     y_size = width * height
#     patches = []

#     if not os.path.exists(yuv_file_path):
#         print(f"Warning: File {yuv_file_path} does not exist.")
#         return [], []

#     with open(yuv_file_path, 'rb') as f:
#         y_data = f.read(y_size)

#     if len(y_data) != y_size:
#         print(f"Warning: Expected {y_size} bytes, got {len(y_data)} bytes.")
#         return [], []

#     y_channel = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))

#     for i in range(0, height, 224):
#         for j in range(0, width, 224):
#             patch = y_channel[i:i+224, j:j+224]
#             if patch.shape[0] < 224 or patch.shape[1] < 224:
#                 patch = np.pad(patch, ((0, 224 - patch.shape[0]), (0, 224 - patch.shape[1])), 'constant')
#             patches.append(patch)
#     return patches


# def load_data_from_csv(csv_path, denoised_dir):
#     df = pd.read_csv(csv_path)
    
#     all_denoised_patches = []
#     all_scores = []
 
#     for _, row in df.iterrows():
#         denoised_file_name = f"denoised_{row['image_name']}.raw"
#         denoised_path = os.path.join(denoised_dir, denoised_file_name)
        
#         denoised_patches = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])
#         all_denoised_patches.extend(denoised_patches)
      
#         patch_scores = row['patch_score'].strip('[]').split(', ')
#         scores = np.array([0 if float(score) == 0 else 1 for score in patch_scores])
      
#         if len(scores) != len(denoised_patches):
#           print(f"Error: Mismatch in number of patches and scores for {row['image_name']}")
#           continue
#         all_scores.extend(scores)

#     return all_denoised_patches, all_scores

# def prepare_data(data, labels):
#     data = np.array(data).astype('float32') / 255.0
#     data = np.expand_dims(data, axis=-1)
#     lbl = np.array(labels)
#     return data, lbl


# denoised_dir = '/ghosting-artifact-metric/dataset/m-gaid-dataset-high-frequency/denoised'
# csv_path = '/ghosting-artifact-metric/Code/WACV/HighFreq/high_frequency_classification_label.csv'

# denoised_patches, labels = load_data_from_csv(csv_path, denoised_dir)
# denoised_patches_np, labels_np = prepare_data(denoised_patches, labels)

# X_minority = denoised_patches_np[labels_np == 1]
# y_minority = labels_np[labels_np == 1]
# X_majority = denoised_patches_np[labels_np == 0]
# y_majority = labels_np[labels_np == 0]


# num_samples_to_generate = len(y_majority) - len(y_minority)


# datagen = ImageDataGenerator(
#     brightness_range=[0.8, 1.2],
#     horizontal_flip=True,
#     vertical_flip=True,
#     rotation_range=15,
# )


# augmented_images = []
# augmented_labels = []


# while len(augmented_labels) < num_samples_to_generate:
#     for x in X_minority:
#         x = x.reshape((1, 224, 224, 1))
#         for x_augmented in datagen.flow(x, batch_size=1):
#             augmented_images.append(x_augmented[0])
#             augmented_labels.append(1)
#             if len(augmented_labels) >= num_samples_to_generate:
#                 break
#         if len(augmented_labels) >= num_samples_to_generate:
#             break


# X_balanced = np.concatenate((X_majority, X_minority, np.array(augmented_images)))
# y_balanced = np.concatenate((y_majority, y_minority, np.array(augmented_labels)))


# X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# class CNNHyperModel(keras_tuner.HyperModel):
#     def build(self, hp):
#         # Define your model architecture using the hyperparameters 'hp'
#         with tf.distribute.MirroredStrategy().scope():
#             model = keras.Sequential()
            
#             # Convolutional layers
#             conv_blocks = hp.Int('conv_blocks', 3, 7, default=5)
#             for i in range(conv_blocks):
#                 filters = hp.Int(f'filters_{i}', 64, 256, step=64, default=128)
#                 kernel_size = hp.Choice(f'kernel_size_{i}', [3, 5], default=3)
#                 if i == 0:
#                     model.add(layers.Conv2D(
#                         filters=filters,
#                         kernel_size=kernel_size,
#                         activation='relu',
#                         padding='same',
#                         input_shape=(224, 224, 1)
#                     ))
#                 else:
#                     model.add(layers.Conv2D(
#                         filters=filters,
#                         kernel_size=kernel_size,
#                         activation='relu',
#                         padding='same'
#                     ))
#                 if hp.Boolean(f'batch_norm_{i}', default=True):
#                     model.add(layers.BatchNormalization())
#                 model.add(layers.MaxPooling2D())
#                 if hp.Boolean(f'conv_dropout_{i}', default=False):
#                     dropout_rate = hp.Float(f'conv_dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.2)
#                     model.add(layers.Dropout(rate=dropout_rate))
            
#             model.add(layers.Flatten())
            
#             # Dense layers
#             dense_blocks = hp.Int('dense_blocks', 1, 4, default=2)
#             for i in range(dense_blocks):
#                 units = hp.Int(f'units_{i}', 64, 512, step=128, default=256)
#                 model.add(layers.Dense(
#                     units=units,
#                     activation='relu'
#                 ))
#                 dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.25)
#                 model.add(layers.Dropout(rate=dropout_rate))
            
#             model.add(layers.Dense(1, activation='sigmoid'))
            
#             # Compile the model
#             model.compile(
#                 optimizer=keras.optimizers.Adam(
#                     learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log', default=1e-3)
#                 ),
#                 loss='binary_crossentropy',
#                 metrics=['accuracy']
#             )
            
#             return model

# # Instantiate the hypermodel
# hypermodel = CNNHyperModel()




# import os
# import json
# import tensorflow as tf
# from sklearn.metrics import accuracy_score




# tuner_dir = '/ghosting-artifact-metric/Artifact-Reduction-Analysis/Byesian_Directory/ghosting_artifact_detection'
# trial_dirs = [os.path.join(tuner_dir, d) for d in os.listdir(tuner_dir) if d.startswith('trial')]

# best_accuracy = 0.0
# best_model = None
# best_hyperparameters = None

# for trial_dir in trial_dirs:
#     trial_json_path = os.path.join(trial_dir, 'trial.json')
#     if os.path.exists(trial_json_path):
#         # Load hyperparameters from trial.json
#         with open(trial_json_path, 'r') as f:
#             trial_data = json.load(f)
#         trial_id = trial_data['trial_id']
#         print(f"Evaluating model from {trial_dir}")

#         # Reconstruct hyperparameters
#         hp = keras_tuner.engine.hyperparameters.HyperParameters()
#         hp_values = trial_data['hyperparameters']['values']
#         for key, value in hp_values.items():
#             hp.Fixed(key, value)

#         # Build the model using the hyperparameters
#         model = hypermodel.build(hp)

#         # Load model weights
#         checkpoints_dir = os.path.join(trial_dir, 'checkpoints')
#         if os.path.exists(checkpoints_dir):
#             latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
#             if latest_checkpoint:
#                 model.load_weights(latest_checkpoint)
#             else:
#                 print(f"No checkpoints found in {checkpoints_dir}")
#                 continue
#         else:
#             print(f"No checkpoints directory in {trial_dir}")
#             continue

#         # Evaluate the model
#         test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#         print(f"Model from {trial_dir} has accuracy: {test_accuracy}")

#         # Keep track of the best model
#         if test_accuracy > best_accuracy:
#             best_accuracy = test_accuracy
#             best_model = model
#             best_hyperparameters = hp

# # After evaluating all models, check if a best model was found
# if best_model is not None:
#     print(f"Best model found with accuracy: {best_accuracy}")
#     # Save the best model
#     # best_model.save('/ghosting-artifact-metric/Artifact-Reduction-Analysis/Byesian_Directory/best_ghosting_artifact_detector.h5')
#     # Optionally, print the best hyperparameters
#     print("Best Hyperparameters:")
#     for key, value in best_hyperparameters.values.items():
#         print(f"{key}: {value}")
# else:
#     print("No valid models were found in the trial directories.")
