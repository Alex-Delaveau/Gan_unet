from models.loss import generator_loss, discriminator_loss
import matplotlib.pyplot as plt

@tf.function
def train_step(generator, discriminator, optimizer_g, optimizer_d, x, y):
    """
    Training step for GAN.
    - x: Input image with mask (masked input).
    - y: Ground truth image (real face).
    """
    with tf.GradientTape(persistent=True) as tape:
        # Generator forward pass
        generated_images = generator(x, training=True)

        # Discriminator forward pass
        real_output = discriminator(y, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Compute losses
        g_loss = generator_loss(fake_output, y, generated_images)
        d_loss = discriminator_loss(real_output, fake_output)

    # Update generator
    gradients_g = tape.gradient(g_loss, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients_g, generator.trainable_variables))

    # Update discriminator
    gradients_d = tape.gradient(d_loss, discriminator.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients_d, discriminator.trainable_variables))

    return g_loss, d_loss


def train(dataset, generator, discriminator, optimizer_g, optimizer_d, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for step, (masked_images, ground_truth) in enumerate(dataset):
            g_loss, d_loss = train_step(generator, discriminator, optimizer_g, optimizer_d, masked_images, ground_truth)

            if step % 100 == 0:
                print(f"Step {step}: Generator Loss: {g_loss.numpy()}, Discriminator Loss: {d_loss.numpy()}")

        # Visualize after each epoch
        sample_visualization(generator, masked_images, ground_truth, epoch)



def sample_visualization(generator, masked_images, ground_truth, epoch):
    """Visualize the results after an epoch."""
    generated_images = generator(masked_images, training=False)

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
    plt.show()
