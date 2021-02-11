import tensorflow as tf


def augment_inversion(image: tf.Tensor) -> tf.Tensor:
    random = tf.random.uniform(shape=[], minval=0, maxval=1)
    if random > 0.5:
        image = tf.math.multiply(image, -1)
        image = tf.math.add(image, 1)
    return image


def augment_rotation(image: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(
        image,
        tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )


def augment_data(image, label):
    image = augment_inversion(image)
    image = augment_rotation(image)
    return image, label
