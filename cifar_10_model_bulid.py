# %% [markdown]
# 📦 CIFAR-10 Projesi
# 1. Veri Setini Hazırama

## 1.1 Kütüphaneler
# %%
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomTranslation,
    RandomRotation,
    Input,
)
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import load_model

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

# import warnings
# warnings.filterwarnings("ignore")

# %% [markdown]
## 1.2 Veri Setinin Yüklenmesi

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)
print("Target:", y_train[:5])

label_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

plt.figure(figsize=(13, 6), dpi=125)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(f"index {i}, Label: {label_map[y_train[i][0]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# %% [markdown]
## 1.3 Normalizasyon
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])


def normalize_data(*datasets, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    return [((data.astype("float32") / 255.0) - mean) / std for data in datasets]


x_train, x_test = normalize_data(x_train, x_test)


# %% [markdown]
## 1.4 One Hot


def one_hot(*datasets, unique_value_count=10):
    return [to_categorical(data, unique_value_count) for data in datasets]

y_train, y_test = one_hot(y_train, y_test)

print("Target:", y_train[:5])


data_augmentation = Sequential([
    RandomFlip("horizontal"),             # Yatay çevirme
    RandomTranslation(0.1, 0.1),          # %10 kaydırma (padding=4 etkisi)
    RandomRotation(0.05),                 # Hafif rotasyon
], name="data_augmentation")

# %% [markdown]
# 2. Ann Modelinin Oluşturulması ve Derlenmesi

wd = 1e-4

model = Sequential([
    Input(shape=(32, 32, 3)),

    data_augmentation,

    Conv2D(64, (3,3), activation="relu", padding="same", kernel_regularizer=l2(wd)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same", kernel_regularizer=l2(wd)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(128, (3,3), activation="relu", padding="same", kernel_regularizer=l2(wd)),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", padding="same", kernel_regularizer=l2(wd)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.3),


    Conv2D(256, (3,3), activation="relu", padding="same", kernel_regularizer=l2(wd)),
    BatchNormalization(),
    Conv2D(256, (3,3), activation="relu", padding="same", kernel_regularizer=l2(wd)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.4),


    GlobalAveragePooling2D(),
    Dense(512, activation="relu", kernel_regularizer=l2(wd)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])

optimizer = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# %% [markdown]
## 2.2 .Callbacks fonksyonlarının tanımlanması ve modelin eğitilmesi

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,         # LR'yi 5'te birine indir
    patience=5,
    min_lr=1e-5,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,         # Daha uzun bekle (LR düşünce kurtulabilir)
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "model/cifar10_best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# %% [markdown]
## 2.3 Callbacks fonksyonlarının tanımlanması ve modelin eğitilmesi

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,   
    callbacks=[reduce_lr, early_stopping, checkpoint]
)



# %% [markdown]
# 3 Modelin Test Edilmesi ve Performansının İncelenmesi

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n🎯 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")



plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Eğitim Kaybı")
plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
plt.title("Eğitim ve Doğrulama Kaybı (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")



plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
plt.plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
plt.title("Eğitim ve Doğrulama Doğruluğu (Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()