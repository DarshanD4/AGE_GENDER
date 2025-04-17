import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, save_model

# === CONFIG ===
DATA_DIR = "Data/UTKFace"
IMG_SIZE = 100
EPOCHS = 10
BATCH_SIZE = 64

# === LOAD DATA ===
def load_data():
    images = []
    ages = []
    genders = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".jpg"):
            parts = filename.split("_")
            try:
                age = int(parts[0])
                gender = int(parts[1])  # 0: Male, 1: Female
                img_path = os.path.join(DATA_DIR, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                ages.append(age)
                genders.append(gender)
            except:
                continue

    X = np.array(images) / 255.0
    y_age = np.array(ages)
    y_gender = to_categorical(np.array(genders), 2)

    return train_test_split(X, y_age, y_gender, test_size=0.2)

# === BUILD MODEL ===
def build_model():
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Gender Output
    gender_output = Dense(2, activation='softmax', name='gender')(x)

    # Age Output
    age_output = Dense(1, activation='linear', name='age')(x)

    model = Model(inputs=input_img, outputs=[age_output, gender_output])
    model.compile(optimizer=Adam(0.001), 
                  loss={'age': 'mse', 'gender': 'categorical_crossentropy'},
                  metrics={'age': 'mae', 'gender': 'accuracy'})
    
    return model

# === MAIN ===
if __name__ == "__main__":
    X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = load_data()
    
    model = build_model()
    model.summary()

    model.fit(X_train, 
              {'age': y_age_train, 'gender': y_gender_train},
              validation_data=(X_test, {'age': y_age_test, 'gender': y_gender_test}),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE)

    os.makedirs("model", exist_ok=True)
    model.save("model/age_gender_model.h5")
