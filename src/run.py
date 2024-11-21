from training.train import train
from models.generator import build_generator
from models.discriminator import build_discriminator

def generate_dataset():
    print("Generating dataset")
    pass

def run_train():
    dataset = generate_dataset()

    generator = build_generator()


    train(dataset=dataset, )



if __name__ == "__main__":
    train()