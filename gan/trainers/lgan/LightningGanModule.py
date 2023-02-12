import os
from einops import rearrange, repeat
import pytorch_lightning as pl

from torch.functional import F
import torch
import torchmetrics
import torchvision


class LightningGanModule(pl.LightningModule):
    def __init__(
        self,
        save_dir: str,
        sample_interval: int,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer,
        aug_types: list[str] = ["translation", "cutout"],
        aug_prob: float = 0.5,
    ):
        super().__init__()
        self.generator = generator
        # self.discriminator = discriminator
        self.D_aug = discriminator  # AugWrapper(self.discriminator)
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.sample_interval = sample_interval

        self.fid = torchmetrics.image.fid.FrechetInceptionDistance(num_features=128)

        self.aug_kwargs = {"prob": aug_prob, "types": aug_types}

    # training step of gan, two optimizers
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        batch_size, channels, height, width = x.shape

        # train generator
        if optimizer_idx == 0:
            # generate fake images

            gen_loss = self.__generator_loss(x)

            if batch_idx % self.sample_interval == 0:
                with torch.no_grad():
                    noise = torch.randn(
                        batch_size, self.generator.latent_dim, device=x.device
                    )
                    generated_images = self.generator(noise)
                    self.__save_training_images(generated_images)

            self.log("loss_g", gen_loss, prog_bar=True)
            return gen_loss

        # train discriminator
        if optimizer_idx == 1:

            disc_loss = self.__discriminator_loss(x)
            self.log("loss_d", disc_loss, prog_bar=True)
            return disc_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size, channels, height, width = x.shape

        # generate fake images
        noise = torch.randn(batch_size, self.generator.latent_dim, device=x.device)
        generated_images = self.generator(noise)
        fake_output, fake_output_32x32, _ = self.D_aug(
            generated_images, detach=True, **self.aug_kwargs
        )

        real_output, real_output_32x32, real_aux_loss = self.D_aug(
            x, calc_aux_loss=True, **self.aug_kwargs
        )

        loss = dual_contrastive_loss(fake_output, real_output)
        loss_32x32 = dual_contrastive_loss(fake_output_32x32, real_output_32x32)

        gen_loss = loss + loss_32x32

        real_output_loss = real_output
        fake_output_loss = fake_output

        divergence = dual_contrastive_loss(real_output_loss, fake_output_loss)
        divergence_32x32 = dual_contrastive_loss(real_output_32x32, fake_output_32x32)
        disc_loss = divergence + divergence_32x32

        aux_loss = real_aux_loss
        disc_loss = disc_loss + aux_loss
        # log loss
        self.log("val_loss_d", disc_loss, prog_bar=True)
        self.log("val_loss_g", gen_loss, prog_bar=True)

        # update fid
        self.__update_fid(generated_images, x)

        # save sample images
        if batch_idx % self.sample_interval == 0:
            self.__save_sample_images(generated_images)

        return {"val_loss_d": disc_loss, "val_loss_g": gen_loss}

    def __save_training_images(self, generated_images):
        save_dir = self.save_dir
        save_path = os.path.join(save_dir, f"last.png")
        generated_images = generated_images.clamp_(0.0, 1.0)
        torchvision.utils.save_image(generated_images, save_path)

    def __save_sample_images(self, generated_images):

        save_dir = self.save_dir
        epoch = self.current_epoch

        save_path = os.path.join(save_dir, f"epoch_{epoch}.png")
        generated_images = generated_images.clamp_(0.0, 1.0)
        torchvision.utils.save_image(generated_images, save_path)

    def validation_epoch_end(self, outputs):
        # log fid
        self.log("fid", self.fid.compute(), prog_bar=True)

    def __update_fid(self, generated_images, real_images):
        # calculate fid

        # scale to [0,1], then denormalize to 255, change to int
        generated_images = (generated_images.clamp_(0.0, 1.0) * 255).to(torch.uint8)

        # denormalize real images to 255, change to int
        real_images = (real_images * 255).to(torch.uint8)

        # calculate fid
        self.fid.update(generated_images, real=False)
        self.fid.update(real_images, real=True)

    def __generator_loss(self, x):
        batch_size = x.shape[0]
        noise = torch.randn(batch_size, self.generator.latent_dim, device=x.device)
        generated_images = self.generator(noise)
        fake_output, fake_output_32x32, _ = self.D_aug(
            generated_images, **self.aug_kwargs
        )
        real_output, real_output_32x32, _ = self.D_aug(x, **self.aug_kwargs)

        loss = dual_contrastive_loss(fake_output, real_output)
        loss_32x32 = dual_contrastive_loss(fake_output_32x32, real_output_32x32)

        gen_loss = loss + loss_32x32

        return gen_loss

    def __discriminator_loss(self, x):
        batch_size = x.shape[0]
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
        divergence_32x32 = dual_contrastive_loss(real_output_32x32, fake_output_32x32)
        disc_loss = divergence + divergence_32x32

        aux_loss = real_aux_loss
        disc_loss = disc_loss + aux_loss
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
        return F.cross_entropy(
            t, torch.zeros(t1.shape[0], device=device, dtype=torch.long)
        )

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)
