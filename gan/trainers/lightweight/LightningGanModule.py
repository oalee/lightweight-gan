from einops import rearrange, repeat
import pytorch_lightning as pl

from torch.functional import F
import torch
from gan.models.lightweight.generator import AugWrapper


class LightningGanModule(pl.LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer,
        aug_types: list[str] = ["translation", "cutout"],
        aug_prob: float = 0.5,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.D_aug = AugWrapper(self.discriminator)
        self.optimizer = optimizer

        self.aug_kwargs = {"prob": aug_prob, "types": aug_types}

    # training step of gan, two optimizers
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        batch_size, channels, height, width = x.shape

        # train generator
        if optimizer_idx == 0:
            # generate fake images
            noise = torch.randn(batch_size, self.generator.latent_dim, device=x.device)
            generated_images = self.generator(noise)
            fake_output, fake_output_32x32, _ = self.D_aug(
                generated_images, **self.aug_kwargs
            )
            real_output, real_output_32x32, _ = self.D_aug(x, **self.aug_kwargs)

            loss = dual_contrastive_loss(fake_output, real_output)
            loss_32x32 = dual_contrastive_loss(fake_output_32x32, real_output_32x32)

            gen_loss = loss + loss_32x32

            gen_loss = gen_loss

            self.log("loss_g", gen_loss, prog_bar=True)
            return gen_loss

        # train discriminator
        if optimizer_idx == 1:
            # generate fake images
            noise = torch.randn(batch_size, self.generator.latent_dim, device=x.device)
            generated_images = self.generator(noise)
            fake_output, fake_output_32x32, _ = self.D_aug(
                generated_images, detach=True, **self.aug_kwargs
            )

            real_output, real_output_32x32, real_aux_loss = self.D_aug(
                x, calc_aux_loss=True, **self.aug_kwargs
            )

            real_output_loss = real_output
            fake_output_loss = fake_output

            divergence = dual_contrastive_loss(real_output_loss, fake_output_loss)
            divergence_32x32 = dual_contrastive_loss(
                real_output_32x32, fake_output_32x32
            )
            disc_loss = divergence + divergence_32x32

            aux_loss = real_aux_loss
            disc_loss = disc_loss + aux_loss
            # log loss
            self.log("loss_d", disc_loss, prog_bar=True)
            return disc_loss

    def configure_optimizers(self):
        return self.optimizer


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(
        lambda t: rearrange(t, "... -> (...)"), (real_logits, fake_logits)
    )

    def loss_half(t1, t2):
        t1 = rearrange(t1, "i -> i ()")
        t2 = repeat(t2, "j -> i j", i=t1.shape[0])
        t = torch.cat((t1, t2), dim=-1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device=device, dtype=torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)
