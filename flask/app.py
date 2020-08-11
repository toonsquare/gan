import os
import sys
from flask import Flask, request, json, make_response
from flask_json import FlaskJSON, JsonError, json_response, as_json
import re
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
import glob

np.set_printoptions(threshold=np.inf)

BATCH_SIZE = 1
BUFFER_SIZE = 1000

IMG_WIDTH = 256
IMG_HEIGHT = 256
LOOP_TO_SAVE = 1

app = Flask(__name__)
json = FlaskJSON(app)

app.debug = True

app.config['JSON_ADD_STATUS'] = False
app.config['JSON_DATETIME_FORMAT'] = '%d/%m/%Y %H:%M:%S'
CORS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'x-requested-with',
    'Access-Control-Allow-Methods': 'GET PUT, DELETE, OPTIONS'
}

OUTPUT_CHANNELS = 3


# %% [code]
def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1

    return input_image


def random_crop(input_image):
    stacked_image = tf.stack([input_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[1, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0]


def downsample(filters, size, shape, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', batch_input_shape=shape,
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, shape, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, batch_input_shape=shape,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def buildGenerator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, (None, 256, 256, 3), apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4, (None, 128, 128, 64)),  # (bs, 64, 64, 128)
        downsample(256, 4, (None, 64, 64, 128)),  # (bs, 32, 32, 256)
        downsample(512, 4, (None, 32, 32, 256)),  # (bs, 16, 16, 512)
        downsample(512, 4, (None, 16, 16, 512)),  # (bs, 8, 8, 512)
        downsample(512, 4, (None, 8, 8, 512)),  # (bs, 4, 4, 512)
        downsample(512, 4, (None, 4, 4, 512)),  # (bs, 2, 2, 512)
        downsample(512, 4, (None, 2, 2, 512)),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, (None, 1, 1, 512), apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, (None, 2, 2, 1024), apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, (None, 4, 4, 1024), apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4, (None, 8, 8, 1024)),  # (bs, 16, 16, 1024)
        upsample(256, 4, (None, 16, 16, 1024)),  # (bs, 32, 32, 512)
        upsample(128, 4, (None, 32, 32, 512)),  # (bs, 64, 64, 256)
        upsample(64, 4, (None, 64, 64, 256)),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generate_images_v2(model, test_input):
    prediction = model(test_input, training=True)
    PredictionImage = prediction.numpy()
    # PredictionImage = list(tf.data.Dataset.as_numpy_iterator(prediction))

    # print(test_input[0].shape)
    # plt.imshow(test_input[0])
    # plt.axis('off')
    # plt.savefig('test_input.png')
    # #
    plt.imshow(PredictionImage[0])
    plt.axis('off')
    plt.savefig('prediction_only.png')

    # todo base64 return
    return "base64"


global generator
generator = buildGenerator()

checkpoint_dir = "../model/Sketch2Color_training_checkpoints_99-100"
# checkpoint_dir = "../model/ckpt-100"

checkpoint = tf.train.Checkpoint(
    generator=generator
)
history = checkpoint.restore(checkpoint_dir)
print(history)


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        app.logger.debug("prediction start")
        upload = request.files['file']
        print(upload.filename)
        upload_data = upload.read()
        # print(upload_data)
        img = Image.open(BytesIO(upload_data))
        npimg = np.array(img)
        npimg = np.reshape(npimg, ((1, 256, 256, 3)))
        npimg = normalize(npimg)

        print(npimg.shape)
        # print(npimg)
        base64 = generate_images_v2(generator, npimg)

        base64 = ""
        return json_response(data_={
            "base64": base64
        }, headers_=CORS)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
