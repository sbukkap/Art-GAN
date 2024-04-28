# Art-GAN
Art restoration in renaissance paintings using Generative Adversarial Networks.

Hello, if you're here, we're assuming you have read our paper report, understood our model framework, and other details.
This readme is to understand the purpose each file/folder in this repostory serves, before running our program.

## Repository Tree
"ArtDataset" contains original "paintings", test_images(cropped), train_images(cropped).

Files named 000generator.pth, 00generator.pth, 0generator.pth, 1generator.pth, 2generator.pth, generator.pth are all saved models, run on different epochs, with different mask sizes,shapes, and varying batches. Our best model is "generator.pth", so we recommend you using that.

Files named 000discriminator.pth, 00discriminator.pth, 0discriminator.pth, 1discriminator.pth, 2discriminator.pth, discriminator.pth are all saved models, run on different epochs, with different mask sizes,shapes, and varying batches. Our best model is "discriminator.pth", so we recommend you using that.

Art_GAN.py is our main python file.

Ignore test.py and Images folder.


## Notes in Art_GAN.py
 - ensure all imported libraries are installed.
 - ensure path is properly set in lines 392,393
 - If you want to test model training, uncomment line 420, save it with a new name in lines 425,426, load it using the same names in lines 430.
 - make sure image_dir and save_dir are properly set in lines 435,436

## How to run ?
(Python version 3.11.5 was used, although earlier versions (unless it's not python2) should work well)
in your terminal, just run "python Art_GAN.py" or "python3 Art_GAN.py"
