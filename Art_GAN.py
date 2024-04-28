from PIL import Image, ImageDraw
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import pickle
from torch.optim import Adam
import torchvision.models as models  # Assuming inception_v4 is available or adapt to a similar model


# GENERATE DEFECT MASK ------------------------------------------------

def generate_defect_mask(image_size, num_shapes=2):
    """
    Generate a defect mask with arbitrary shapes.

    Parameters:
    - image_size: tuple of (width, height) for the mask.
    - num_shapes: number of shapes to draw on the mask. Randomly chosen between 4 to 7 if not specified.

    Returns:
    - A PIL.Image object representing the mask.
    """
    mask = Image.new('L', image_size, color='white')  # 'L' mode for grayscale
    draw = ImageDraw.Draw(mask)

    shape_functions = [
        lambda: draw.point([random.randint(0, image_size[0]), random.randint(0, image_size[1])], fill='black'),
        lambda: draw.line([random.randint(0, image_size[0]), random.randint(0, image_size[1]),
                           random.randint(0, image_size[0]), random.randint(0, image_size[1])], fill='black', width=15),
        lambda: draw.rectangle(generate_rect_coords(image_size), outline='black', width=15),
        lambda: draw.ellipse(generate_ellipse_coords(image_size), outline='black', width=15)
    ]

    for _ in range(random.randint(num_shapes, num_shapes+3)):
        random.choice(shape_functions)()

    # Convert to numpy array and normalize to 0-1
    mask_np = np.array(mask) / 255.0

    return Image.fromarray(np.uint8(mask_np * 255), 'L')  # Convert back to PIL Image

def generate_rect_coords(image_size):
    x0 = random.randint(0, image_size[0])
    y0 = random.randint(0, image_size[1])
    x1 = random.randint(x0, image_size[0])
    y1 = random.randint(y0, image_size[1])
    return [x0, y0, x1, y1]

def generate_ellipse_coords(image_size):
    x0 = random.randint(0, image_size[0])
    y0 = random.randint(0, image_size[1])
    x1 = random.randint(x0, image_size[0])
    y1 = random.randint(y0, image_size[1])
    return [x0, y0, x1, y1]


# APPLY MASK TO IMAGE

# To apply this mask to an image (assuming 'image' is a PIL Image):
def apply_mask(image, mask):
    image_np = np.array(image)
    mask_np = np.array(mask) / 255.0  # Normalize mask to 0-1
    defect_image_np = image_np * mask_np[:,:,None]  # Apply mask to each channel
    return Image.fromarray(np.uint8(defect_image_np))

# GENERATOR CLASS ------------------------------------------------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Dedicated down-sampling blocks
        self.down1a = self.conv_bn_relu(3, 64, 5, 1, 2)
        self.down1b = self.conv_bn_relu(64, 128, 3, 2, 1)
        self.down2a = self.conv_bn_relu(128, 128, 3, 1, 1)
        self.down2b = self.conv_bn_relu(128, 256, 3, 2, 1)
        self.down3a = self.conv_bn_relu(256, 256, 3, 1, 1)
        self.down3b = self.conv_bn_relu(256, 256, 3, 2, 1)

        # Dilated convolutions
        self.dilated_conv1 = self.dilated_conv(256, 256, dilation=2)
        self.dilated_conv2 = self.dilated_conv(256, 256, dilation=4)
        self.dilated_conv3 = self.dilated_conv(256, 256, dilation=8)
        self.dilated_conv4 = self.dilated_conv(256, 256, dilation=16)

        # Up-sampling
        self.up1a = self.conv_bn_relu(256, 256, 3, 1, 1)
        self.up1b = self.conv_bn_relu(320, 256, 3, 1, 1)
        self.up2 = self.conv_bn_relu(192, 128, 3, 1, 1)
        self.up3 = self.conv_bn_relu(96, 64, 3, 1, 1)


        # Outputs
        self.out_content = self.output_img(64, 3) # Account for the concatenated channels
        self.out_line = self.output_sketch(64, 1)   # Account for the concatenated channels

    def conv_bn_relu(self, in_channels, out_channels, k_size, s, p):
        # Additional conv-bn-relu applied after concatenation and pixel shuffle
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=s, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def dilated_conv(self, in_channels, out_channels, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def output_img(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def output_sketch(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Down-sampling
        d1_skip = self.down1a(x)
        d1 = self.down1b(d1_skip)
        d2_skip = self.down2a(d1)
        d2 = self.down2b(d2_skip)
        d3_skip = self.down3a(d2)
        d3 = self.down3b(d3_skip)

        # Dilated convolutions
        dilated = self.dilated_conv1(d3)
        dilated = self.dilated_conv2(dilated)
        dilated = self.dilated_conv3(dilated)
        dilated = self.dilated_conv4(dilated)

        # Up-sampling
        u1 = self.up1a(dilated)
        u1 = self.pixel_shuffle(u1)  # Pixel shuffle
        u1 = torch.cat((u1, d3_skip), 1) # Concatenation
        u1 = self.up1b(u1)

        u2 = self.pixel_shuffle(u1)  # Pixel shuffle
        u2 = torch.cat((u2, d2_skip), 1)  # Concatenation
        u2 = self.up2(u2)  # Additional Conv-BN-ReLU

        u3 = self.pixel_shuffle(u2)  # Pixel shuffle
        u3 = torch.cat((u3, d1_skip), 1)  # Concatenation
        u3 = self.up3(u3)

        # Output
        content = self.out_content(u3)
        line = self.out_line(u3)

        return content, line

    def pixel_shuffle(self, x):
        # Apply the pixel shuffle operation here
        return nn.PixelShuffle(2)(x)

# DISCRIMINATOR CLASS ------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.i_model = nn.Sequential(
            self.conv_bn_relu(3, 64, 4, 2, 1),
            self.conv_bn_relu(64, 128, 4, 2, 1),
            self.conv_bn_relu(128, 256, 4, 2, 1),
            self.conv_bn_relu(256, 256, 4, 2, 1),
            self.conv_bn_relu(256, 256, 4, 2, 1)
        )

        self.s_model = nn.Sequential(
            self.conv_bn_relu(1, 64, 4, 2, 1),
            self.conv_bn_relu(64, 128, 4, 2, 1),
            self.conv_bn_relu(128, 256, 4, 2, 1),
            self.conv_bn_relu(256, 256, 4, 2, 1),
            self.conv_bn_relu(256, 256, 4, 2, 1)
        )

        self.final_conv = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Add global average pooling
        self.sigmoid = nn.Sigmoid()

    def conv_bn_relu(self, in_channels, out_channels, k_size, s, p):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=s, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img, sketch):
        img_features = self.i_model(img)
        sketch_features = self.s_model(sketch)
        combined_features = torch.cat((img_features, sketch_features), dim=1)
        combined_features = self.final_conv(combined_features)
        combined_features = self.global_pool(combined_features)  # Pool to reduce to 1x1
        decision = self.sigmoid(combined_features.view(combined_features.size(0), -1))
        return decision

# GENERATE SKETCH ------------------------------------------------

# Define a function to generate a sketch using edge detection
def generate_sketch(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = np.invert(edges)
    return Image.fromarray(edges)

# ART DATASET CLASS ------------------------------------------------

class ArtDataset(Dataset):
    def __init__(self, image_dir, image_transform=None, mask_transform=None, sketch_transform=None, train=True):
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.sketch_transform = sketch_transform
        if train:
            self.image_filenames = [f for f in os.listdir(image_dir) if f.startswith('train')]
        else:
            self.image_filenames = [f for f in os.listdir(image_dir) if f.startswith('test')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        mask = generate_defect_mask(image.size)
        sketch = generate_sketch(image)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.functional.to_tensor(mask)

        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)
        else:
            sketch = transforms.functional.to_tensor(sketch)

        # Apply mask to simulate defect
        defect_image = image * mask

        return defect_image, mask, image, sketch

# TRAINING ------------------------------------------------

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_v3 = models.inception_v3(pretrained=True, progress=True)
        self.inception_v3.eval()
        for param in self.inception_v3.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        # Assumption: output and target are preprocessed images
        output_features = self.inception_v3(output)
        target_features = self.inception_v3(target)
        loss = F.l1_loss(output_features, target_features)
        return loss


# def train_stage(generator, discriminator, dataloader, epochs, optimizer_G, optimizer_D, criterion_GAN, criterion_MSE, stage):
#     for epoch in range(epochs):
#         for i, (defect_images, masks, real_images, sketches) in enumerate(dataloader):
#             defect_images = defect_images.to(device)
#             masks = masks.to(device)
#             real_images = real_images.to(device)
#             sketches = sketches.to(device)
#             valid = torch.ones(defect_images.size(0), 1, device=defect_images.device)
#             fake = torch.zeros(defect_images.size(0), 1, device=defect_images.device)

#             # Generate images and sketches
#             gen_images, gen_sketches = generator(defect_images)

#             # ---------------------
#             #  Train Discriminator
#             # ---------------------
#             if stage == 'both' or stage == 'D_only':
#                 optimizer_D.zero_grad()
#                 real_loss = criterion_GAN(discriminator(real_images, sketches), valid)
#                 fake_loss = criterion_GAN(discriminator(gen_images.detach(), gen_sketches.detach()), fake)
#                 d_loss = (real_loss + fake_loss) / 2
#                 d_loss.backward()
#                 optimizer_D.step()

#             if stage =='G_only':
#                 optimizer_G.zero_grad()
#                 # Generator loss from discriminator feedback
#                 # feedback = discriminator(gen_images, gen_sketches)
#                 # g_loss_GAN = criterion_GAN(feedback, valid)
#                 # MSE loss for image quality
#                 g_loss_MSE = criterion_MSE(gen_images, real_images)
#                 g_loss = g_loss_MSE
#                 g_loss.backward()
#                 optimizer_G.step()

#             # -----------------
#             #  Train Generator
#             # -----------------
#             if stage == 'both':
#                 optimizer_G.zero_grad()
#                 # Generator loss from discriminator feedback
#                 feedback = discriminator(gen_images, gen_sketches)
#                 g_loss_GAN = criterion_GAN(feedback, valid)
#                 # MSE loss for image quality
#                 g_loss_MSE = criterion_MSE(gen_images, real_images)
#                 g_loss = g_loss_GAN + g_loss_MSE
#                 g_loss.backward()
#                 optimizer_G.step()

#             # Prepare loss reports for printing
#             d_loss_report = d_loss.item() if stage != 'G_only' else 'N/A'
#             g_loss_report = g_loss.item() if stage != 'D_only' else 'N/A'
#             print(f"[Stage {stage}] [Epoch {epoch}/{epochs}] [Batch {i+1}/{len(dataloader)}] [D loss: {d_loss_report}] [G loss: {g_loss_report}]")

# def staged_training(generator, discriminator, dataloader, epochs, optimizer_G, optimizer_D, criterion_GAN, criterion_MSE):
#     # Stage 1: Train generator only
#     print('Stage 1: Train generator only')
#     train_stage(generator, discriminator, dataloader, epochs, optimizer_G, optimizer_D, criterion_GAN, criterion_MSE, 'G_only')
#     # Stage 2: Train discriminator only
#     print('Stage 2: Train discriminator only')
#     train_stage(generator, discriminator, dataloader, epochs, optimizer_G, optimizer_D, criterion_GAN, criterion_MSE, 'D_only')
#     # Stage 3: Train both
#     print('Stage 3: Train both')
#     train_stage(generator, discriminator, dataloader, epochs, optimizer_G, optimizer_D, criterion_GAN, criterion_MSE, 'both')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(generator, discriminator, dataloader, epochs, optimizer_G, optimizer_D, lambda_mse, lambda_perc, lambda_gan):
    generator.to(device)
    discriminator.to(device)
    criterion_MSE = nn.MSELoss().to(device)
    criterion_perceptual = PerceptualLoss().to(device)

    for epoch in range(epochs):
        for i, (defect_images, masks, real_images, sketches) in enumerate(dataloader):
            real_images = real_images.to(device)
            defect_images = defect_images.to(device)
            sketches = sketches.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_preds = discriminator(real_images, sketches)
            fake_images, fake_sketches = generator(defect_images)
            fake_preds = discriminator(fake_images.detach(), fake_sketches.detach())
            d_loss = -(torch.mean(real_preds) - torch.mean(fake_preds))
            d_loss.backward()
            optimizer_D.step()

            # Clipping weights for Wasserstein GAN
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_images, fake_sketches = generator(defect_images)
            gen_preds = discriminator(gen_images, fake_sketches)
            mse_loss = criterion_MSE(gen_images, real_images)
            perceptual_loss = criterion_perceptual(gen_images, real_images)
            g_loss = -torch.mean(gen_preds) + lambda_mse * mse_loss + lambda_perc * perceptual_loss
            g_loss.backward()
            optimizer_G.step()

            if i % 50 == 0:  # Printing every 50 batches
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# SET UP TRAINING ------------------------------------------------
# PATH = '/Users/tylerrowe/Desktop/College/24Spring/Computer Vision/Project/GAN/'
PATH = './'
image_dir = f'{PATH}ArtDataset/train_images'
# Set up the dataloader, models, and training
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.6137, 0.5450, 0.4393], std=[0.2425, 0.2519, 0.2569])
])

art_dataset = ArtDataset(image_dir, image_transform=image_transform)

dataloader = DataLoader(art_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion_GAN = nn.BCELoss().to(device)
criterion_MSE = nn.MSELoss().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

num_epochs = 100

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

train(generator, discriminator, dataloader, num_epochs, optimizer_G, optimizer_D, 1.0, 0.1, 0.01)



# TRAINING ------------------------------------------------

# staged_training(generator, discriminator, dataloader, num_epochs, optimizer_G, optimizer_D, criterion_GAN, criterion_MSE)

# Save the models
# torch.save(generator.state_dict(), f'{PATH}generator.pth')
# torch.save(discriminator.state_dict(), f'{PATH}discriminator.pth')

# # Load the generator model
# generator = Generator()
# generator.load_state_dict(torch.load('./generator.pth', map_location=torch.device('cpu')))
# generator.eval()  # Set the model to evaluation mode

# # Assuming the device is properly set
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Assuming the dataset and dataloader are properly defined
# # Assuming the image_transform is the same as used during training
# image_dir = f'{PATH}ArtDataset/test_images'
# save_dir = "./ArtDataset/testRes"
# os.makedirs(save_dir, exist_ok=True)


# image_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.6137, 0.5450, 0.4393], std=[0.2425, 0.2519, 0.2569])
# ])
# art_dataset = ArtDataset(image_dir, image_transform=image_transform, train=False)
# dataloader = DataLoader(art_dataset, batch_size=16, shuffle=True)

# # Generate some masked and completed images
# data_iter = iter(dataloader)
# masked_images, masks, real_images, sketches = next(data_iter)
# masked_images = masked_images.to(device)
# completed_images, _ = generator(masked_images)

# # Move tensors to CPU and convert to numpy for visualization
# masked_images = masked_images.cpu().detach()
# real_images = real_images.cpu().detach()
# completed_images = completed_images.cpu().detach()
# sketches = sketches.cpu().detach()

# # Define the denormalization transform
# def denormalize(tensors):
#     means = np.array([0.6137, 0.5450, 0.4393])
#     stds = np.array([0.2425, 0.2519, 0.2569])
#     means = means.reshape((1, 3, 1, 1))
#     stds = stds.reshape((1, 3, 1, 1))
#     tensors = tensors * torch.tensor(stds) + torch.tensor(means)
#     return tensors.clamp(0, 1)  # Clamp to the range [0, 1]

# # Denormalize images
# masked_images = denormalize(masked_images)
# completed_images = denormalize(completed_images)
# real_images = denormalize(real_images)
# sketches = denormalize(sketches)

# # Convert tensors to numpy arrays for plotting
# masked_images = masked_images.numpy().transpose((0, 2, 3, 1))
# completed_images = completed_images.numpy().transpose((0, 2, 3, 1))
# real_images = real_images.numpy().transpose((0,2,3,1))
# sketches = sketches.numpy().transpose((0,2,3,1))

# # Plot the images
# num_images = 2  # Number of images to display
# fig, axes = plt.subplots(num_images, 4, figsize=(15, 25))

# # Set headings for each column
# axes[0, 0].set_title('Masked Image', fontsize=12)
# axes[0, 1].set_title('Sketch', fontsize=12)
# axes[0, 2].set_title('Completed Image', fontsize=12)
# axes[0, 3].set_title('Real Image', fontsize=12)

# for i in range(num_images):
#     # Display masked image
#     axes[i, 0].imshow(masked_images[i])
#     axes[i, 0].axis('off')

#     # Display sketch
#     axes[i, 1].imshow(sketches[i])
#     axes[i, 1].axis('off')

#     # Display completed image
#     axes[i, 2].imshow(completed_images[i])
#     axes[i, 2].axis('off')

#     # Display real image
#     axes[i, 3].imshow(real_images[i])
#     axes[i, 3].axis('off')

# plt.tight_layout(pad=3.0)
# plt.show()

# import cv2
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# # Assuming real_images and completed_images are numpy arrays containing the ground truth and generated images respectively

# psnr_scores = []
# ssim_scores = []

# mse_loss = np.mean((real_images - completed_images) ** 2)

# print(len(real_images), len(completed_images))
# for i in range(len(real_images)):
#     # Convert images to uint8 format (required by PSNR and SSIM functions)
#     real_image_uint8 = (real_images[i] * 255).astype('uint8')
#     completed_image_uint8 = (completed_images[i] * 255).astype('uint8')

#     # Calculate PSNR
#     psnr_score = psnr(real_image_uint8, completed_image_uint8)

#     # Calculate SSIM with an explicit window size
#     window_size = 3 # You can adjust this value as needed
#     ssim_score = ssim(real_image_uint8, completed_image_uint8, win_size=window_size, multichannel=True)

#     psnr_scores.append(psnr_score)
#     ssim_scores.append(ssim_score)

# # Compute average PSNR and SSIM scores
# average_psnr = sum(psnr_scores) / len(psnr_scores)
# average_ssim = sum(ssim_scores) / len(ssim_scores)

# print("Average MSE:", mse_loss)
# print("Average PSNR:", average_psnr)
# print("Average SSIM:", average_ssim)





# # Load one sample from the dataloader
# defect_image, mask, real_image, sketch = next(iter(dataloader))

# # Move data to the device
# defect_image = defect_image.to(device)

# # Generate an image using the generator
# with torch.no_grad():
#     generated_image, _ = generator(defect_image)

# # Convert tensors to PIL images
# defect_image_pil = transforms.ToPILImage()(defect_image.cpu().squeeze())
# real_image_pil = transforms.ToPILImage()(real_image.cpu().squeeze())

# # Convert generated image tensor to numpy array and remove normalization
# generated_image_np = generated_image.cpu().squeeze().permute(1, 2, 0).numpy()
# generated_image_np = (generated_image_np * 0.5) + 0.5  # De-normalize the generated image
# generated_image_pil = Image.fromarray((generated_image_np * 255).astype('uint8'))  # Convert numpy array back to PIL Image

# # Plot the images
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.title('Masked Image')
# plt.imshow(defect_image_pil)
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title('Generated Image')
# plt.imshow(generated_image_pil)
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.title('Ground Truth Image')
# plt.imshow(real_image_pil)
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# image_dir = './ArtDataset/train_images'
# # Set up the dataloader, models, and training
# image_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.6137, 0.5450, 0.4393], std=[0.2425, 0.2519, 0.2569])
# ])

# art_dataset = ArtDataset(image_dir, image_transform=image_transform)

# # Save art_dataset locally
# with open('./art_dataset.pkl', 'wb') as f:
#     pickle.dump(art_dataset, f)

# dataloader = DataLoader(art_dataset, batch_size=16, shuffle=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generator = Generator().to(device)
# discriminator = Discriminator().to(device)

# criterion_GAN = nn.BCELoss().to(device)
# criterion_MSE = nn.MSELoss().to(device)

# optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
# optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

# staged_training(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion_GAN, criterion_MSE)
# torch.save(generator.state_dict(), './generator.pth')
# torch.save(discriminator.state_dict(), './discriminator.pth')

# # Function to denormalize images
# def denormalize(tensor):
#     tensor = tensor.clone()  # Avoid modifying tensor in-place
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)  # Reverse the normalization
#     return tensor

# # Fetch one batch of data
# data_iter = iter(dataloader)
# defect_images, masks, real_images, sketches = next(data_iter)

# # Forward pass through the generator
# line_images, completed_images = generator(defect_images)

# # Convert images to the appropriate format for visualization
# defect_images = defect_images.cpu()
# masks = masks.cpu()
# real_images = real_images.cpu()
# line_images = line_images.cpu()
# completed_images = completed_images.cpu()

# # Plotting
# fig, axs = plt.subplots(4, 4, figsize=(15, 10))  # Adjust the subplot grid as needed

# for i in range(4):
#     # Show masked images
#     masked_img = (defect_images[i] * masks[i]).permute(1, 2, 0)
#     axs[0, i].imshow(masked_img)
#     axs[0, i].axis('off')
#     axs[0, i].set_title('Masked Image')

#     # Show line images from generator
#     line_img = line_images[i].squeeze(0)
#     axs[1, i].imshow(line_img, cmap='gray')
#     axs[1, i].axis('off')
#     axs[1, i].set_title('Line Image')

#     # Show completed images from generator
#     completed_img = denormalize(completed_images[i]).permute(1, 2, 0)
#     axs[2, i].imshow(completed_img)
#     axs[2, i].axis('off')
#     axs[2, i].set_title('Completed Image')

#     # Show original images (ground truth)
#     real_img = denormalize(real_images[i]).permute(1, 2, 0)
#     axs[3, i].imshow(real_img)
#     axs[3, i].axis('off')
#     axs[3, i].set_title('Original Image')

# plt.show()

# line_images, completed_images = generator(defect_images)

# # Generate some masked and completed images
# data_iter = iter(dataloader)
# masked_images, masks, real_images, sketches = next(data_iter)
# masked_images = masked_images.to(device)
# completed_images, _ = generator(masked_images)

# # Convert tensors to numpy arrays and denormalize
# masked_images = masked_images.cpu().detach().numpy()
# completed_images = completed_images.cpu().detach().numpy()

# # Denormalize images
# masked_images = np.transpose(masked_images, (0, 2, 3, 1))  # Change from (N, C, H, W) to (N, H, W, C)
# completed_images = np.transpose(completed_images, (0, 2, 3, 1))

# # Plot the images
# num_images = 5  # Number of images to display
# fig, axes = plt.subplots(num_images, 2, figsize=(10, 20))

# for i in range(num_images):
#     # Display masked image
#     axes[i, 0].imshow(masked_images[i])
#     axes[i, 0].set_title('Masked Image')
#     axes[i, 0].axis('off')

#     # Display completed image
#     axes[i, 1].imshow(completed_images[i])
#     axes[i, 1].set_title('Completed Image')
#     axes[i, 1].axis('off')

# plt.tight_layout()
# plt.show()

# def denormalize(tensor):
#     mean = [0.6137, 0.5450, 0.4393]
#     std = [0.2425, 0.2519, 0.2569]
#     for i in range(3):
#         tensor[i] = tensor[i] * std[i] + mean[i]
#     return tensor

# # Fetch one batch of data
# data_iter = iter(dataloader)
# defect_images, masks, real_images, sketches = next(data_iter)

# # Convert images to the appropriate format for visualization
# defect_images = defect_images.cpu()
# masks = masks.cpu()
# real_images = real_images.cpu()
# line_images = line_images.cpu()
# completed_images = completed_images.cpu()
