from training.train import train
from models.generator import build_generator
from models.discriminator import build_discriminator
from data.dataset import split_dataset
from config.config import Config
import tensorflow as tf

def generate_dataset():
    print("Generating dataset")
    train_dataset = split_dataset(Config.KAGGLE_PATH, Config.BATCH_SIZE)

    return train_dataset

def get_optimizer():
    print("Getting optimizer")
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=Config.GENERATOR_LEARNING_RATE, beta_1=Config.BETA_1)
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=Config.DISCRIMINATOR_LEARNING_RATE, beta_1=Config.BETA_1)
    return optimizer_g, optimizer_d


def print_dataset_info(dataset):
    print("Dataset Info")
    print(dataset)

def run_train():
    train_dataset = generate_dataset()    

    generator = build_generator(input_shape=Config.INPUT_SHAPE)
    discriminator = build_discriminator(input_shape=Config.INPUT_SHAPE)

    optimizer_g, optimizer_d = get_optimizer()

    train(dataset=train_dataset, generator=generator, discriminator=discriminator, optimizer_g=optimizer_g, optimizer_d=optimizer_d, epochs=Config.EPOCHS)



if __name__ == "__main__":
    run_train()