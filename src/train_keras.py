import os
import keras
import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dropout
from keras import models
from keras import layers
import pandas as pd
from pathlib import Path

model = keras.applications.vgg16.VGG16()
conv_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=[224, 224, 3],
)


dropout = 0.4
epoch = 100
batch_size = 64


def get_img(df):
    cols = ["img_path", "filename"]
    df["fullpath"] = f"{Path.home()}" + df[cols].apply(
        lambda row: "/".join(row.values.astype(str)), axis=1
    )
    img_path = df.fullpath.values.tolist()

    scores = df.mouse_score.values.tolist()
    img_label = []
    for i in scores:
        if i == 2.5 or i == 2.0:
            # print(img_label)
            img_label.append(0)
        elif i == 3.0 or i == 3.5:
            img_label.append(1)
        elif i == 4.0 or i == 4.5:
            img_label.append(2)
        elif i == 5.0:
            img_label.append(3)
        else:
            print("Wrong label")

    return img_path, img_label


def load_and_preprocess_image(path, label):
    image_string = tf.compat.as_str_any(path)
    image_string = tf.io.read_file(path)
    img = tf.io.decode_png(image_string, channels=3)
    img_width, img_height = 224, 224
    img_label = tf.one_hot(label, 4)
    return tf.image.resize(img, [img_width, img_height]), img_label


def load_and_preprocess_from_path_labels(path, label):
    return load_and_preprocess_image(path, label)


if __name__ == "__main__":
    main_path = f"{Path.home()}/Data/Sample/"

    # getting the csv files
    ext = "csv"
    count = 1
    df = []
    for subdir, dirs, files in os.walk(main_path + "train"):
        for i in files:
            if i.endswith(ext):
                csv_path = subdir + "/" + i
                csv_pd = pd.read_csv(csv_path)
                if count == 1:
                    df = csv_pd
                elif count > 1:
                    df = pd.concat([df, csv_pd])
                count += 1
    train_df = df

    ext = "csv"
    count = 1
    df = []
    for subdir, dirs, files in os.walk(main_path + "test"):
        for i in files:
            if i.endswith(ext):
                csv_path = subdir + "/" + i
                csv_pd = pd.read_csv(csv_path)
                if count == 1:
                    df = csv_pd
                elif count > 1:
                    df = pd.concat([df, csv_pd])
                count += 1
    test_df = df

    # get the images and labels
    train_path, train_label = get_img(train_df)
    test_path, test_label = get_img(test_df)

    # data loader to tensor
    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_label))
    train_dataset = train_ds.map(load_and_preprocess_from_path_labels)
    train_dataset = train_dataset.batch(64)
    train_dataset = train_dataset.shuffle(buffer_size=10, seed=74)

    test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_label))
    test_dataset = test_ds.map(load_and_preprocess_from_path_labels)
    test_dataset = test_dataset.batch(10)

    # build the model
    model = models.Sequential()
    model.add(conv_base)
    model.add(Dropout(dropout))
    model.add(layers.Flatten())
    model.add(Dropout(dropout))
    model.add(layers.Dense(128, activation="relu"))
    model.add(Dropout(dropout))
    model.add(layers.Dense(4, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # training
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epoch,
        batch_size=batch_size,
        shuffle=True,
    )

    score = model.evaluate(test_dataset)
    # score accuracy
    print(score[1] * 100)

    # convert the model to tfjs
    tfjs.converters.save_keras_model(model, f"{Path.home()}/Data/Sample/Result/")

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_resnet_5fold.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model_resnet_5fold.h5")
    print("Saved model to disk")
