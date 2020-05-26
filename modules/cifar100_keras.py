import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import cifar100

import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple

Shape = namedtuple("Size", ["rows", "columns"])


def convert_rbg_to_grayscale(data: np.ndarray) -> np.ndarray:
    return 0.3 * data[:, :, :, 0] + 0.59 * data[:, :, :, 1] + 0.11 * data[:, :, :, 2]


def reshape(data: np.ndarray, shape=None) -> np.ndarray:
    """
    Each image is 1-dimensional array.
    This function reshapes each image to 2-dimensional matrix,
        where its size is defined with `shape`.
        
    Parameters
    ----------
    data: Data that must be reshaped
    shape: Target shape of each image. 
        If not specified, it will reshape to Shape(28, 28)
    
    Returns
    -------
    np.ndarray
    """
    if shape is None:
        shape = Shape(32, 32)

    reshaped_data = data.reshape(data.shape[0], shape.rows, shape.columns, 1)
    return reshaped_data


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize data:
        - convert all values in a range within 0 and 1
        - cast all values to float
        
    Parameters
    ----------
    data: Data for training
    
    Returns
    -------
    np.ndarray
    """
    return data.astype("float32") / 255


def encode_into_onehot(y: np.ndarray, number_classes: int):
    """
    Convert an array of answers into onehot encoded matrix.
    
    Parameters
    ----------
    y: target array that will be convert to one hot encoded matrix
    number_classes: total number of classes
    
    Returns
    -------
    np.ndarray, matrix with shape of (number_classes x number_classes)
    """

    if number_classes < 1:
        raise ValueError(
            f"Number of classes must be greater than 1, given {number_classes}"
        )

    one_hot_encoded_y = keras.utils.to_categorical(y, number_classes)
    return one_hot_encoded_y


def visualize_accuracy(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy of the model")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def visualize_loss(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss of the model")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def build_keras_model(
    window_size: Shape, pool_size: Shape, input_size: tuple, number_classes: int
):
    """
    Build keras model based on Sequential
    
    Parameters
    ----------
    window_size: the sliding size of the pixels grid that is convolved. 
    pool_size: Window size which extract features.
    number_classes: Total number of classes that must be distinguished.
    
    Returns
    -------
    Keras model which can be fitted and evaluated.
    """
    model = keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_size))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(
        Dense(512, kernel_initializer="lecun_normal", activation="selu", use_bias=True)
    )
    model.add(
        Dense(512, kernel_initializer="lecun_normal", activation="selu", use_bias=True)
    )
    model.add(Dropout(0.5))
    model.add(Dense(units=number_classes, activation="softmax"))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    (xtrain, ytrain), (xtest, ytest) = cifar100.load_data()
    print("Train data:", xtrain.shape, ytrain.shape)
    print("Test data:", xtest.shape, ytest.shape)

    converted_xtrain = convert_rbg_to_grayscale(xtrain)
    converted_xtest = convert_rbg_to_grayscale(xtest)

    reshaped_xtrain = reshape(converted_xtrain, shape=Shape(32, 32))
    reshaped_xtest = reshape(converted_xtest, shape=Shape(32, 32))

    valid_xtrain = normalize(reshaped_xtrain)
    valid_xtest = normalize(reshaped_xtest)

    valid_ytrain = encode_into_onehot(ytrain, number_classes=100)
    valid_ytest = encode_into_onehot(ytest, number_classes=100)

    print("Xtrain shape: ", valid_xtrain.shape)
    print("Ytrain shape: ", valid_ytrain.shape)
    print("Xtest shape: ", valid_xtest.shape)
    print("Ytest shape: ", valid_ytest.shape)

    model = build_keras_model(
        window_size=Shape(5, 5),
        pool_size=Shape(2, 2),
        input_size=(32, 32, 1),
        number_classes=100,
    )

    history = model.fit(
        valid_xtrain,
        valid_ytrain,
        batch_size=128,
        epochs=50,
        validation_data=(valid_xtest, valid_ytest),
        verbose=1,
    )

    loss, accuracy = model.evaluate(valid_xtest, valid_ytest, verbose=0)
    print("Test loss:", loss * 100)
    print("Test accuracy:", accuracy * 100)

    visualize_accuracy(history)
    visualize_loss(history)
