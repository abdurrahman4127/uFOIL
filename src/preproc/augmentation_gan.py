import torch
import torch.nn as nn


class Generator(nn.Module):
    #latent vector in, 256x256 image out.
    def __init__(self, latent_dim: int = 100, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    # 256x256 image in, real/fake score out.
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 8, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img).view(-1)


def train_step(
    generator: Generator,
    discriminator: Discriminator,
    real_batch: torch.Tensor,
    gen_optim: torch.optim.Optimizer,
    disc_optim: torch.optim.Optimizer,
    device: torch.device,
    latent_dim: int = 100,
) -> tuple[float, float]:
    """
    one adversarial training step on a batch of real images.

    args:
        generator: generator network.
        discriminator: discriminator network.
        real_batch: batch of real images, shape (B, C, H, W), in [-1, 1].
        gen_optim: optimizer for the generator.
        disc_optim: optimizer for the discriminator.
        latent_dim: size of the generator's input noise vector.

    returns:
        tuple of (discriminator_loss, generator_loss) for this step.
    """
    batch_size = real_batch.size(0)
    criterion = nn.BCELoss()

    real_labels = torch.ones(batch_size, device=device)
    fake_labels = torch.zeros(batch_size, device=device)

    # discriminator step
    disc_optim.zero_grad()

    real_preds = discriminator(real_batch)
    disc_loss_real = criterion(real_preds, real_labels)

    noise = torch.randn(batch_size, latent_dim, device=device)
    fake_batch = generator(noise)
    fake_preds = discriminator(fake_batch.detach())
    disc_loss_fake = criterion(fake_preds, fake_labels)

    disc_loss = disc_loss_real + disc_loss_fake
    disc_loss.backward()
    disc_optim.step()

    # generator step, wants discriminator to call its fakes real
    gen_optim.zero_grad()
    fake_preds = discriminator(fake_batch)
    gen_loss = criterion(fake_preds, real_labels)
    gen_loss.backward()
    gen_optim.step()

    return float(disc_loss.item()), float(gen_loss.item())


def generate_synthetic_batch(
    generator: Generator,
    n_samples: int,
    device: torch.device,
    latent_dim: int = 100,
) -> torch.Tensor:
    """
    sampling a batch of synthetic images from a trained generator.
    batch of generated images, shape (n_samples, C, H, W), in [-1, 1].
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim, device=device)
        samples = generator(noise)
    generator.train()
    return samples