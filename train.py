import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import json

# import wandb

torch.backends.cudnn.benchmark = True


def train_fn(
        disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    total_g_loss_train = 0
    total_d_real_loss_train = 0
    total_d_fake_loss_train = 0
    total_d_loss_train = 0
    d_fake = 0
    d_real = 0
    g_loss = 0

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        total_g_loss_train += G_loss
        total_d_real_loss_train += D_real_loss.item()
        total_d_fake_loss_train += D_fake_loss.item()
        total_d_loss_train += D_loss.item()

        if idx % 10 == 0:
            d_real = torch.sigmoid(D_real).mean().item()
            d_fake = torch.sigmoid(D_fake).mean().item()
            loop.set_postfix(
                D_real=d_real,
                D_fake=d_fake,
            )
        g_loss = G_loss

    g_loss_train = total_g_loss_train / len(loader)
    d_real_loss_train = total_d_real_loss_train / len(loader)
    d_fake_loss_train = total_d_fake_loss_train / len(loader)
    d_loss_train = total_d_loss_train / len(loader)

    return g_loss_train, d_real_loss_train, d_fake_loss_train, d_loss_train, d_real, d_fake, g_loss


def write_list(a_list, name):
    # print("Started writing list data into a json file")
    with open(name + ".json", "w") as fp:
        json.dump(a_list, fp)
        # print("Done writing JSON data into .json file")


def main(num_residuals):
    g_loss_train = {}
    d_real_loss_train = {}
    d_fake_loss_train = {}
    d_loss_train = {}
    d_real = {}
    d_fake = {}
    g_loss = {}

    # wandb.login()
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(img_channels=3, num_features=64, num_residuals=num_residuals).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999), )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # print(gen)
    additional_dir = ""
    if config.IS_COLAB:
        additional_dir += "/content/gdrive/MyDrive/resnet25block/"

    if config.LOAD_MODEL:
        load_checkpoint(
            additional_dir + config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            additional_dir + config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    '''
    my_config = dict(
        epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        dataset="Contour Drawing",
        architecture="ResNet")
    '''

    # wandb.init(project="DoodleIt", config=my_config)
    # access all HPs through wandb.config, so logging matches execution!
    # my_config = wandb.config

    for epoch in range(0, config.NUM_EPOCHS):
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        # wandb.watch(gen,log="all", log_freq=2)
        # wandb.watch(disc,log="all", log_freq=2)

        g_loss_train[epoch], d_real_loss_train[epoch], d_fake_loss_train[epoch], d_loss_train[epoch], d_real[epoch], \
        d_fake[epoch], g_loss[epoch] = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        g_loss_train1 = list(g_loss_train)
        d_real_loss_train1 = list(d_real_loss_train)
        d_fake_loss_train1 = list(d_fake_loss_train)
        d_loss_train1 = list(d_loss_train)
        d_real1 = list(d_real)
        d_fake1 = list(d_fake)
        g_loss1 = list(g_loss)

        g_loss_train[epoch] = float(g_loss_train[epoch])
        d_real_loss_train[epoch] = float(d_real_loss_train[epoch])
        d_fake_loss_train[epoch] = float(d_fake_loss_train[epoch])
        d_loss_train[epoch] = float(d_loss_train[epoch])
        d_real[epoch] = float(d_real[epoch])
        d_fake[epoch] = float(d_fake[epoch])
        g_loss[epoch] = float(g_loss[epoch])

        write_list(g_loss_train, additional_dir + "g_loss_train")
        write_list(d_real_loss_train, additional_dir + "d_real_loss_train")
        write_list(d_fake_loss_train, additional_dir + "d_fake_loss_train")
        write_list(d_loss_train, additional_dir + "d_loss_train")
        write_list(d_real, additional_dir + "d_real")
        write_list(d_fake, additional_dir + "d_fake")
        write_list(g_loss, additional_dir + "g_loss")

        save_checkpoint(gen, opt_gen, filename=additional_dir + f"{epoch}" + config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename=additional_dir + f"{epoch}" + config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader,
                           epoch,
                           folder=additional_dir + "evaluation")


if __name__ == "__main__":
    main(25)
