import torch
from torch.nn.functional import adaptive_avg_pool2d
import scipy as sp

from torchmetrics import StructuralSimilarityIndexMeasure,PeakSignalNoiseRatio,ErrorRelativeGlobalDimensionlessSynthesis,MeanSquaredError
import torch
import numpy as np



import config
from torchvision.utils import save_image

#import torchviz
#import wandb

from torchmetrics.functional.classification import binary_precision


def calculate_activation_statistics(images, model, batch_size=config.BATCH_SIZE, dims=2048,
                                    cuda=False):
    model.eval()
    act = np.empty((len(images), dims))

    if cuda:
        batch = images.cuda()
    else:
        batch = images
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = sp.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sp.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fretchet(images_real, images_fake, model):
    mu_1, std_1 = calculate_activation_statistics(images_real, model)
    mu_2, std_2 = calculate_activation_statistics(images_fake, model)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value

def calculate_SSIM(preds,target):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(config.DEVICE)
    return ssim(preds, target)

def calculate_PSNR(preds,target):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(config.DEVICE)
    return psnr(preds, target)

def calculate_ergas(preds,target):
    ergas = ErrorRelativeGlobalDimensionlessSynthesis().to(config.DEVICE)
    return torch.round(ergas(preds, target))

def calculate_mse(preds,target):
    mse = MeanSquaredError().to(config.DEVICE)
    return mse(preds,target)


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()




def save_test_examples(gen, val_loader, epoch, folder):
    x = next(iter(val_loader))
    x = x.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")

        '''
        # some evaluations
        fretchet_dist = calculate_fretchet(y, y_fake, gen)
        ssim = calculate_SSIM(y, y_fake)
        psnr = calculate_PSNR(y, y_fake)
        ergas = calculate_ergas(y, y_fake)
        mse = calculate_mse(y, y_fake)

        print(f"FID: {fretchet_dist}")
        print(f"SSIM: {ssim}")
        print(f"PSNR: {psnr}")
        print(f"ergas: {ergas}")
        print(f"mse: {mse}")
        '''
        '''
        # Visualize the generator architecture
        generator_output = y_fake
        torchviz.make_dot(gen(x), params=dict(gen.named_parameters())).render("generator", format="png")

        # Visualize the discriminator architecture
        discriminator_output = disc(generator_output)
        torchviz.make_dot(discriminator_output, params=dict(disc.named_parameters())).render("discriminator", format="png")
        '''
    gen.train()

    #torch.onnx.export(gen, x, "model.onnx")
    #wandb.save("model.onnx")


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
