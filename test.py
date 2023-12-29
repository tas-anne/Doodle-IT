import json

import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_test_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import TestDataset, MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
def write_list(a_list,name):
    print("Started writing list data into a json file")
    with open(name + ".json", "w") as fp:
        json.dump(a_list, fp)
        print("Done writing JSON data into .json file")

def main(num_residuals):
    '''
    wandb.login(key="1a07c0d84e9e88bb2035ff14597b230dd7af542f")
    my_config = dict(
        epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        dataset="Contour Drawing",
        architecture="ResNet")

    wandb.init(project="DoodleIt Test", config=my_config)
    # access all HPs through wandb.config, so logging matches execution!
    my_config = wandb.config
    '''

    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(img_channels=3, num_features=64, num_residuals=num_residuals).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    GEN_DIR = "Resnet with 25 block/20gen.pth.tar"
    DISC_DIR = "Resnet with 25 block/20disc.pth.tar"

    load_checkpoint(
        GEN_DIR, gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        DISC_DIR, disc, opt_disc, config.LEARNING_RATE,
    )

    # print(gen)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    test_dataset = MapDataset(root_dir=config.TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    fretchet_dist = {}
    ssim = {}
    psnr = {}
    ergas = {}
    mse = {}
    print(gen)
    print("RRRRRRRRRRRRRRRRRRRRRRRR")
    print(disc)
    for epoch in range(0, 1):
        print(epoch)
        fretchet_dist[epoch], ssim[epoch], psnr[epoch], ergas[epoch], mse[epoch] = save_some_examples(gen, test_loader,
                                                                                                      epoch,
                                                                                                      folder="evaluation")

        fretchet_dist[epoch]=float(fretchet_dist[epoch])
        ssim[epoch]=float(ssim[epoch])
        psnr[epoch]=float(psnr[epoch])
        ergas[epoch]=float(ergas[epoch])
        mse[epoch]=float(mse[epoch])

        fig, axs = plt.subplots(4, 4)

        fretchet_dist1 = list(fretchet_dist)
        ssim1 = list(ssim)
        psnr1 = list(psnr)
        ergas1 = list(ergas)
        mse1 = list(mse)


        write_list(fretchet_dist,"fretchet_dist")
        write_list(ssim,"ssim1")
        write_list(psnr,"psnr1")
        write_list(ergas,"ergas1")
        write_list(mse,"mse1")

        axs[0, 0].plot(range(0, epoch +1), fretchet_dist1)
        axs[0, 0].set_title('Fretchet Inception Distance')
        axs[0, 0].set(xlabel='Epochs', ylabel='FID')

        axs[2, 1].plot(range(0, epoch +1), ssim1)
        axs[2, 1].set_title('Structural Similarity Index Measure')
        axs[2, 1].set(xlabel='Epochs', ylabel='SSIM')

        axs[2, 2].plot(range(0, epoch +1), psnr1)
        axs[2, 2].set_title('Peak Signal to Noise Ratio')
        axs[2, 2].set(xlabel='Epochs', ylabel='PSNR')

        axs[3, 0].plot(range(0, epoch +1), ergas1)
        axs[3, 0].set_title('Error Relative Global Dim. Synthesis ')
        axs[3, 0].set(xlabel='Epochs', ylabel='ERGAS')

        axs[3, 1].plot(range(0, epoch +1), mse1)
        axs[3, 1].set_title('Mean Squared Error')
        axs[3, 1].set(xlabel='Epochs', ylabel='MSE')

        fig.tight_layout()
        fig.savefig('plot.png')


if __name__ == "__main__":
    main(25)
