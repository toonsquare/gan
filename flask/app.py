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
import base64



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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


# preprocess data
def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1

    return input_image

def random_crop(input_image):
    stacked_image = tf.stack([input_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[1, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0]

#for using generator model
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
    prediction = model(test_input, training=False)
    PredictionImage = prediction.numpy()
    PredictionImage = PredictionImage[0]

    #converting
    img = Image.fromarray(img.astype("uint8"))

    buff =BytesIO()
    img.save(buff, format = "PNG")
    buff.seek(0)
    base64_img = base64.b64encode(buff.getvalue()).decode("utf-8")

    return base64_img


global generator
generator = buildGenerator()

checkpoint_dir = "../model_pt/ckpt-100"
checkpoint = tf.train.Checkpoint(
    generator=generator
)
checkpoint.restore(checkpoint_dir)
generator.save('cmodel.h5')
new_model = tf.keras.models.load_model('cmodel.h5')

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        app.logger.debug("prediction start")
        upload = request.files['file']
        print(upload.filename)
        upload_data = upload.read()

        img = Image.open(BytesIO(upload_data))
        npimg = np.array(img)
        npimg = np.reshape(npimg, ((1, 256, 256, 3)))
        npimg = normalize(npimg)

        test_jpg = tf.io.read_file('../test4.jpg')
        test_jpg = tf.image.decode_jpeg(test_jpg)
        # test_jpg = tf.image.resize(test_jpg, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        test_jpg = tf.cast(test_jpg, tf.float32) / 127.5 - 1
        test_jpg = tf.reshape(test_jpg, [1, 256, 256, 3])
        print(test_jpg.shape)
        result = generate_images_v2(new_model, test_jpg)

#       result = ""
        return json_response(data_={
            "result": result
        }, headers_=CORS)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
