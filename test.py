import matplotlib.pyplot as plt
import numpy as np



# Generate some masked and completed images
data_iter = iter(dataloader)
masked_images, masks, real_images, sketches = next(data_iter)
masked_images = masked_images.to(device)
completed_images, _ = generator(masked_images)

# Convert tensors to numpy arrays and denormalize
masked_images = masked_images.cpu().detach().numpy()
completed_images = completed_images.cpu().detach().numpy()

# Denormalize images
masked_images = np.transpose(masked_images, (0, 2, 3, 1))  # Change from (N, C, H, W) to (N, H, W, C)
completed_images = np.transpose(completed_images, (0, 2, 3, 1))

# Plot the images
num_images = 5  # Number of images to display
fig, axes = plt.subplots(num_images, 2, figsize=(10, 20))

for i in range(num_images):
    # Display masked image
    axes[i, 0].imshow(masked_images[i])
    axes[i, 0].set_title('Masked Image')
    axes[i, 0].axis('off')

    # Display completed image
    axes[i, 1].imshow(completed_images[i])
    axes[i, 1].set_title('Completed Image')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()