#unzipping the dataset
import zipfile

with zipfile.ZipFile("brain-tumor-mri-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

# Import necessary libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set image size and path
IMG_SIZE = 128
DATA_DIR = "data/Training"
CATEGORIES = ["glioma" , "meningioma" , "notumor" , "pituitary"]

X = []
y = []

# Go through all image files in the dataset or LOAD AND PREPROCESS ALL THE IMAGES
for category in CATEGORIES:
    folder_path = os.path.join(DATA_DIR, category)
    
    # Label from filename: 'tumor (...)' -> 1, 'no (...)' -> 0
    label = 0 if category == "notumor" else 1
    
    for img_name in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
        except Exception as e:
            pass

# Convert to NumPy arrays
X = np.array(X) / 255.0  # Normalize pixel values
y = to_categorical(np.array(y), num_classes=2)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset loaded and preprocessed successfully.")
print(f"Total images: {len(X)}, Tumor: {np.sum(np.argmax(y, axis=1))}, No Tumor: {len(y)-np.sum(np.argmax(y, axis=1))}")

#build CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(2, activation='softmax')  # 2 classes: tumor or no tumor
])

#solving overfittting issue
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen.fit(X_train)


Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(IMG_SIZE, IMG_SIZE, 3))

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,                     # You can increase epochs since early stopping handles it
    batch_size=32,
    callbacks=[early_stop]         # ← Add this
)


# Save the model
model.save("brain_tumor_model.h5")

print("Model training completed and saved successfully.")







