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

BATCH_SIZE = 1
BUFFER_SIZE = 1000

IMG_WIDTH = 256
IMG_HEIGHT = 256
LOOP_TO_SAVE = 1

app = Flask(__name__)
json = FlaskJSON(app)

val_file = glob.glob('./data/val/*.png')

app.debug = True

app.config['JSON_ADD_STATUS'] = False
app.config['JSON_DATETIME_FORMAT'] = '%d/%m/%Y %H:%M:%S'
CORS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'x-requested-with',
    'Access-Control-Allow-Methods': 'GET PUT, DELETE, OPTIONS'
}

OUTPUT_CHANNELS = 3

checkpoint_dir = "./model/ckpt-100"
checkpoint = tf.train.Checkpoint()
checkpoint.restore(checkpoint_dir)


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     256, 256)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


# %% [code]
def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# %% [code]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load(image_file):
    image = tf.io.read_file(image_file)
    # image = image_file
    image = tf.image.decode_jpeg(image, channels=3)

    w = tf.shape(image)[1]
    w = w // 2

    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     256, 256)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


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


@tf.function()
def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)
    print(prediction[0])
    print(type(prediction[0].numpy()))
    # img = Image.fromarray(prediction[0])
    # img.save("./prediction.jpg")
    # prediction[0] = tf.image.convert_image_dtype(prediction[0], dtype=tf.uint8)
    # tf.io.encode_jpeg(prediction[0])
    # print(prediction[0].shape)
    # print(type(prediction[0]))
    # plt.figure(figsize=(15, 15))
    #
    # display_list = [test_input[0], tar[0], prediction[0]]
    # title = ['Input Image', 'Ground Truth', 'Predicted Image']
    #
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.title(title[i])
    #     plt.imshow(display_list[i])
    #     plt.axis('off')
    #
    # plt.savefig('image_at_epoch_{:04d}.png'.format('prediction'))


test_dataset = tf.data.Dataset.from_tensor_slices(val_file)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

global generator
generator = buildGenerator()


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        app.logger.debug("prediction start")
        upload = request.files['file']
        img = Image.open(upload)
        # sketch, target = load_image_test(np.array(img))
        #
        # sketch = np.array([sketch])
        # target = np.array([target])
        # # app.logger.debug(content['input'])
        # print(sketch.shape)
        # print(target.shape)
        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)
        # generate_images(generator, sketch, target)

        # results = {"prediction" : content['input']}

        return json_response(data_={}, headers_=CORS)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
