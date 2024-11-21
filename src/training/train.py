from models.loss import generator_loss, discriminator_loss
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tqdm import tqdm
from config.config import Config

@tf.function
def train_discriminator_step(discriminator, optimizer_d, real_images, generated_images):
    """
    Train the discriminator for one step.
    """
    with tf.GradientTape() as tape:
        # Discriminator forward pass
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Compute discriminator loss
        d_loss = discriminator_loss(real_output, fake_output)

    # Update discriminator
    gradients_d = tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients_d, discriminator.trainable_variables))

    return d_loss


@tf.function
def train_generator_step(generator, discriminator, optimizer_g, x, y):
    """
    Train the generator for one step.
    """
    with tf.GradientTape() as tape:
        # Generator forward pass
        generated_images = generator(x, training=True)

        # Discriminator forward pass on generated images
        fake_output = discriminator(generated_images, training=True)

        # Compute generator loss
        g_loss = generator_loss(fake_output, y, generated_images)

    # Update generator
    gradients_g = tape.gradient(g_loss, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients_g, generator.trainable_variables))

    return g_loss



def train(dataset, generator, discriminator, optimizer_g, optimizer_d, epochs):

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Add tqdm for step progress
        epoch_progress = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch + 1}")

        for step, (masked_images, ground_truth) in epoch_progress:
            for _ in range(2):  # Train discriminator twice per generator step
                generated_images = generator(masked_images, training=True)
                d_loss = train_discriminator_step(discriminator, optimizer_d, ground_truth, generated_images)

            g_loss = train_generator_step(generator, discriminator, optimizer_g, masked_images, ground_truth)

            # Update tqdm description with loss values
            epoch_progress.set_postfix({
                "G_Loss": f"{g_loss.numpy():.4f}",
                "D_Loss": f"{d_loss.numpy():.4f}"
            })
            if step % 100 == 0:
                print(f"Step {step}: Generator Loss: {g_loss.numpy()}, Discriminator Loss: {d_loss.numpy()}")
                sample_visualization(generator, masked_images, ground_truth, epoch)

        save_models(generator, discriminator, epoch, output_dir=Config.OUTPUT_DIR)  

        # Visualize after each epoch

        


def save_models(generator, discriminator, epoch, output_dir="saved_models"):
    """
    Save the generator and discriminator models.
    
    Parameters:
    - generator: The generator model to save.
    - discriminator: The discriminator model to save.
    - epoch: Current epoch number.
    - output_dir: Directory to save the models.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the generator
    generator.save(os.path.join(output_dir, f"generator_epoch_{epoch + 1}.h5"))
    # Save the discriminator
    discriminator.save(os.path.join(output_dir, f"discriminator_epoch_{epoch + 1}.h5"))

    print(f"Models saved for epoch {epoch + 1} to {output_dir}")



def sample_visualization(generator, masked_images, ground_truth, epoch, output_dir="visualizations"):
    """
    Visualize and save the results after an epoch.
    - generator: The trained generator model.
    - masked_images: Input images with masks.
    - ground_truth: Ground truth images.
    - epoch: Current epoch number.
    - output_dir: Directory to save the visualization files.
    """
    # Generate images
    generated_images = generator(masked_images, training=False)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the file name
    file_name = f"epoch_{epoch:03d}.png"
    file_path = os.path.join(output_dir, file_name)

    # Create the visualization
    plt.figure(figsize=(15, 5))
    for i in range(3):  # Show 3 samples
        plt.subplot(3, 3, i * 3 + 1)
        plt.title("Masked Input")
        plt.imshow(masked_images[i].numpy())
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 2)
        plt.title("Ground Truth")
        plt.imshow(ground_truth[i].numpy())
        plt.axis("off")

        plt.subplot(3, 3, i * 3 + 3)
        plt.title("Generated Output")
        plt.imshow(generated_images[i].numpy())
        plt.axis("off")

    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(file_path)
    plt.close()  # Close the figure to free memory

    print(f"Visualization for epoch {epoch} saved to {file_path}")

