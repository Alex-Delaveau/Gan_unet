import tensorflow as tf

def reconstruction_loss(y_true, y_pred):
    """L1 loss for reconstruction."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def adversarial_loss(fake_likelihood):
    """Adversarial loss for generator."""
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_likelihood), logits=fake_likelihood))


def generator_loss(disc_fake_output, y_true, y_pred, lambda_recon=10):
    """
    Generator loss = reconstruction loss + adversarial loss.
    lambda_recon is a weight to emphasize reconstruction.
    """
    recon_loss = reconstruction_loss(y_true, y_pred)
    adv_loss = adversarial_loss(disc_fake_output)
    return lambda_recon * recon_loss + adv_loss


def discriminator_loss(real_output, fake_output):
    """
    Discriminator loss = real loss + fake loss.
    - real_output: discriminator predictions for real images.
    - fake_output: discriminator predictions for generator outputs.
    """
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

