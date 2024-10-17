import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator():
    model = models.Sequential()

    model.add(layers.Dense(256 * 32 * 32, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((32, 32, 256)))

    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(16, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=1, padding='same', use_bias=False, activation='tanh'))

    return model

def compile_generator(generator):
    generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator

def generate_images(generator, noise):
    generated_images = generator(noise, training=False)
    return generated_images

def train_generator(generator, discriminator, epochs, batch_size, noise_dim, real_images):
    for epoch in range(epochs):
        for batch in range(0, real_images.shape[0], batch_size):
            noise = tf.random.normal([batch_size, noise_dim])

            generated_images = generator(noise, training=True)
            real_images_batch = real_images[batch:batch + batch_size]

            labels_real = tf.ones((batch_size, 1))
            labels_fake = tf.zeros((batch_size, 1))

            # Combine real and fake images for training the discriminator
            combined_images = tf.concat([real_images_batch, generated_images], axis=0)
            combined_labels = tf.concat([labels_real, labels_fake], axis=0)

            with tf.GradientTape() as disc_tape:
                predictions = discriminator(combined_images, training=True)
                disc_loss = tf.keras.losses.binary_crossentropy(combined_labels, predictions)

            grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            with tf.GradientTape() as gen_tape:
                generated_images = generator(noise, training=True)
                predictions = discriminator(generated_images, training=True)
                gen_loss = tf.keras.losses.binary_crossentropy(labels_real, predictions)

            grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
            generator.optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Generator Loss: {gen_loss.numpy().mean()}, Discriminator Loss: {disc_loss.numpy().mean()}')

def build_discriminator():
    model = models.Sequential()

    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(256, 256, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def compile_discriminator(discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator


def load_real_images(image_directory):
    image_filenames = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    images = []
    for filename in image_filenames:
        img_path = os.path.join(image_directory, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))  # Resize images to 256x256
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1] range
        images.append(img)
    return np.array(images)


def main():
    noise_dim = 100
    epochs = 200
    batch_size = 32

    generator = build_generator()
    discriminator = build_discriminator()

    compile_generator(generator)
    compile_discriminator(discriminator)

    # Load your real images dataset here
    image_directory = 'content/dataset'  # Update this path
    real_images = load_real_images(image_directory)

    train_generator(generator, discriminator, epochs, batch_size, noise_dim, real_images)

if __name__ == '__main__':
    main()
