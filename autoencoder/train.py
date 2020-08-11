from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import matplotlib.pyplot as plt
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 256
IMG_HEIGHT = 256
LOOP_TO_SAVE = 1
EPOCHS = 50
BATCH_SIZE = 200
BUFFER_SIZE = 100

# MNIST데이터 셋을 불러옵니다.

(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

image_file = glob.glob('../data/train/*.png')
val_file = glob.glob('../data/val/*.png')

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True,
                                   width_shift_range=False,
                                   height_shift_range=False,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    '../data/train',
    target_size=(256, 256),
    batch_size=50
)

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  horizontal_flip=True,
                                  width_shift_range=False,
                                  height_shift_range=False,
                                  fill_mode='nearest')
test_generator = test_datagen.flow_from_directory(
    '../data/val',
    target_size=(256, 256),
    batch_size=50
)


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)

    w = tf.shape(image)[1]
    w = w // 2

    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# %% [code]
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


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     256, 256)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


@tf.function
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


train_dataset = tf.data.Dataset.from_tensor_slices(image_file)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(val_file)
test_dataset = test_dataset.map(load_image_test)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# auto encoder
autoencoder = Sequential()
# autoencoder.add(Conv2D(16, kernel_size=3, padding='same', input_shape=(256, 256, 3), activation='relu'))
autoencoder.add(Conv2D(16, kernel_size=3, padding='same', input_shape=(28, 28, 1), activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
autoencoder.add(Conv2D(8, kernel_size=3, activation='relu', padding='same'))
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
autoencoder.add(Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu'))

# decoding
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(16, kernel_size=3, activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))

autoencoder.summary()


# 컴파일 및 학습을 하는 부분입니다.
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=test_generator,
                                    validation_steps=4)

# Todo 출력 부분 구현


# working code for mnist
# autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_test, X_test))
#
# # 학습된 결과를 출력하는 부분입니다.
# random_test = np.random.randint(X_test.shape[0], size=5)  # 테스트할 이미지를 랜덤하게 불러옵니다.
# ae_imgs = autoencoder.predict(X_test)  # 앞서 만든 오토인코더 모델에 집어 넣습니다.
#
# plt.figure(figsize=(7, 2))  # 출력될 이미지의 크기를 정합니다.
#
# for i, image_idx in enumerate(random_test):  # 랜덤하게 뽑은 이미지를 차례로 나열합니다.
#     ax = plt.subplot(2, 7, i + 1)
#     plt.imshow(X_test[image_idx].reshape(28, 28))  # 테스트할 이미지를 먼저 그대로 보여줍니다.
#     ax.axis('off')
#     ax = plt.subplot(2, 7, 7 + i + 1)
#     plt.imshow(ae_imgs[image_idx].reshape(28, 28))  # 오토인코딩 결과를 다음열에 출력합니다.
#     ax.axis('off')
# plt.savefig("autoencoder_test.jpg")
