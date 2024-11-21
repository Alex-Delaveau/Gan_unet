from tensorflow.keras import layers, models

def build_discriminator(input_shape):
    """Builds a discriminator model."""
    input_img = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(input_img)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)  # Real or fake

    return models.Model(input_img, x)
