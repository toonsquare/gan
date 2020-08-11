# %% [code]
import tensorflow as tf
import os
import glob
import time
from matplotlib import pyplot as plt

# %% [code]
image_file = glob.glob('./data/train/*.png')

# %% [code]
IMG_WIDTH = 256
IMG_HEIGHT = 256
LOOP_TO_SAVE = 1

# builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('./model')
# sess = tf.compat.v1.InteractiveSession()

generated_image_dir = './generated/'
checkpoint_dir = './models/Sketch2Color_training_checkpoints'
last_checkpoint_dir = './model/Sketch2Color_training_checkpoints_99'
checkpoint = tf.train.Checkpoint()


# %% [code]
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


# %% [code]
@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


# %% [code]
def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# %% [code]
def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     256, 256)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# %% [code]
BATCH_SIZE = 100
BUFFER_SIZE = 1000

# %% [code]
train_dataset = tf.data.Dataset.from_tensor_slices(image_file)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# %% [code]
val_file = glob.glob('./data/val/*.png')

# %% [code]
test_dataset = tf.data.Dataset.from_tensor_slices(val_file)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

# %% [code]
OUTPUT_CHANNELS = 3


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

    result.add(tf.keras.layers.LeakyReLU())

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


# %% [code]
generator = buildGenerator()

# %% [code]
LAMBDA = 10


# %% [code]
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# %% [code]
def downs(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def buildDiscriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downs(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downs(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downs(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# %% [code]
discriminator = buildDiscriminator()

# %% [code]
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# %% [code]
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


# %% [code]
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

# %% [code]
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# %% [code]
def generate_images(model, test_input, tar, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig(generated_image_dir + 'image_at_epoch_{:04d}.png'.format(epoch))


# %% [code]
@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


# %% [code]
def fit(train_ds, epochs, test_ds):
    print("BATCH : " + str(BATCH_SIZE) + " EPOCH :" + str(EPOCHS))
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target, epoch)

        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)

        print()

        checkpoint_prefix = os.path.join(checkpoint_dir + "_" + str(epoch))
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    # checkpoint.save(file_prefix = checkpoint_prefix)


# %% [code]
EPOCHS = 100

# %% [code]
fit(train_dataset, EPOCHS, test_dataset)

# print("restore ckpt")
# # %% [code]
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# #
# for example_input, example_target in test_dataset.take(5):
#     print(example_input.shape)
#     print(example_target.shape)
#     generate_images(generator, example_input, example_target)

# generator.save('ColorizationModelv1.h5')

# %% [code]
