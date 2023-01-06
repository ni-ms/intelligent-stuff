import numpy as np
from tensorflow import keras, expand_dims
from tensorflow.keras import layers

num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
try:
    model = keras.models.load_model("existing_model")
except IOError:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
batch_size = 64
epochs = 25
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save("existing_model")

print(model.summary())
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

img = keras.utils.load_img("sample1.png").resize((28, 28)).convert("L")
img = keras.utils.img_to_array(img)
img = img.reshape(1, 28, 28, 1)
img = img.astype("float32") / 255
score = model.predict(img)
print(score)
print("The number is", np.argmax(score))
print("The accuracy is", np.max(score) * 100)
