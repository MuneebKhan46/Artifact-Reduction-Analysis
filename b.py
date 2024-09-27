
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras_tuner  
from keras_tuner import HyperModel, Hyperband, BayesianOptimization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


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
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
)


augmented_images = []
augmented_labels = []


while len(augmented_labels) < num_samples_to_generate:
    for x in X_minority:
        x = x.reshape((1, 224, 224, 1))
        for x_augmented in datagen.flow(x, batch_size=1):
            augmented_images.append(x_augmented[0])
            augmented_labels.append(1)
            if len(augmented_labels) >= num_samples_to_generate:
                break
        if len(augmented_labels) >= num_samples_to_generate:
            break


X_balanced = np.concatenate((X_majority, X_minority, np.array(augmented_images)))
y_balanced = np.concatenate((y_majority, y_minority, np.array(augmented_labels)))


X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class CNNHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        with strategy.scope():
            model = keras.Sequential()
            
            # Convolutional layers
            conv_blocks = hp.Int('conv_blocks', 3, 7, default=5)
            for i in range(conv_blocks):
                filters = hp.Int(f'filters_{i}', 64, 256, step=64, default=128)
                kernel_size = hp.Choice(f'kernel_size_{i}', [3, 5], default=3)
                if i == 0:
                    model.add(layers.Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        padding='same',
                        input_shape=(224, 224, 1)
                    ))
                else:
                    model.add(layers.Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        padding='same'
                    ))
                if hp.Boolean(f'batch_norm_{i}', default=True):
                    model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling2D())
                if hp.Boolean(f'conv_dropout_{i}', default=False):
                    dropout_rate = hp.Float(f'conv_dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.2)
                    model.add(layers.Dropout(rate=dropout_rate))
            
            model.add(layers.Flatten())
            
            # Dense layers
            dense_blocks = hp.Int('dense_blocks', 1, 4, default=2)
            for i in range(dense_blocks):
                units = hp.Int(f'units_{i}', 64, 512, step=128, default=256)
                model.add(layers.Dense(
                    units=units,
                    activation='relu'
                ))
                dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.25)
                model.add(layers.Dropout(rate=dropout_rate))
            
            model.add(layers.Dense(1, activation='sigmoid'))
            
            # Compile the model
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log', default=1e-3)
                ),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model

hypermodel = CNNHyperModel()

tuner = BayesianOptimization(
    hypermodel,
    objective='val_accuracy',
    max_trials=100, 
    directory='/ghosting-artifact-metric/Artifact-Reduction-Analysis/Byesian_Directory',
    project_name='ghosting_artifact_detection',
    distribution_strategy=strategy
)

# early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

# tuner.search( X_train, y_train, epochs=50, validation_split=0.15, callbacks=[early_stopping], verbose=1)

trial_id = '24'  
specific_trial = tuner.oracle.trials[trial_id]
specific_trial.hyperparameters['model']

specific_hps = specific_trial.hyperparameters
specific_model = tuner.hypermodel.build(specific_hps)

# Compile the model (if needed)
specific_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test data
test_loss, test_accuracy = specific_model.evaluate(X_test, y_test)
print(f'Test Accuracy for trial #20: {test_accuracy * 100:.2f}%')

# Make predictions
y_pred_probs = specific_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_probs)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Print the hyperparameters used in trial #20
print("Hyperparameters for trial #20:")
for key, value in specific_hps.values.items():
    print(f"{key}: {value}")

# Rebuild the model for a new input shape if necessary (e.g., (224, 224, 1))
specific_model.build(input_shape=(None, 224, 224, 1))
specific_model.summary()





# # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# # best_model = tuner.hypermodel.build(best_hps)

# best_models = tuner.get_best_models(num_models=1)
# best_model = best_models[0]
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
# print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# y_pred_probs = best_model.predict(X_test)
# y_pred = (y_pred_probs > 0.5).astype("int32")

# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# roc_auc = roc_auc_score(y_test, y_pred_probs)
# print(f'ROC AUC Score: {roc_auc:.2f}')




# print("Best Hyperparameters:")
# for key, value in best_hps.values.items():
#     print(f"{key}: {value}")



# best_model.build(input_shape=(None, 224, 224, 1))
# best_model.summary()



# # best_model.save('/ghosting-artifact-metric/Artifact-Reduction-Analysis/Byesian_Directory/best_ghosting_artifact_detector.h5')
