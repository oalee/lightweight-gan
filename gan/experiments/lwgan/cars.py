from ...data.cars import CarsLightningDataModule, DiffAugment, AugWrapper
from ...trainers.lgan import LightningGanModule
from ...models.lgan import Generator, Discriminator
from torch import nn
import yerbamate, torch, pytorch_lightning as pl, pytorch_lightning.callbacks as pl_callbacks, os

env = yerbamate.Environment()

data_module = CarsLightningDataModule(
    image_size=128,
    aug_prob=0.5,
    in_channels=3,
    data_dir=env["data_dir"],
    batch_size=8,
)

generator = Generator(
    image_size=128,
    latent_dim=128,
    fmap_max=256,
    fmap_inverse_coef=12,
    transparent=False,
    greyscale=False,
    attn_res_layers=[],
    freq_chan_attn=False,
    norm_class=nn.BatchNorm2d,
)

discriminator = Discriminator(
    image_size=128,
    fmap_max=256,
    fmap_inverse_coef=12,
    transparent=False,
    greyscale=False,
    disc_output_size=5,
    attn_res_layers=[],  # [16, 32, 64, 128, 256],
)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
)

g_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    g_optimizer, mode="min", factor=0.5, patience=5, verbose=True
)
d_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    d_optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

g_optimizer = {
    "optimizer": g_optimizer,
    "lr_scheduler": {"scheduler": g_lr_scheduler, "monitor": "fid"},
}

d_optimizer = {
    "optimizer": d_optimizer,
    "lr_scheduler": {"scheduler": d_lr_scheduler, "monitor": "fid"},
}

aug_types = ["translation", "cutout", "color", "offset"]

model = LightningGanModule(
    save_dir=env["results"],
    sample_interval=100,
    generator=generator,
    discriminator=AugWrapper(discriminator),
    optimizer=[g_optimizer, d_optimizer],
    aug_types=aug_types,
    aug_prob=0.5,
)

logger = pl.loggers.TensorBoardLogger(env["results"], name=env.name)

callbacks = [
    pl_callbacks.ModelCheckpoint(
        monitor="fid",
        dirpath=env["results"],
        save_top_k=1,
        mode="min",
        save_last=True,
    ),
    pl_callbacks.LearningRateMonitor(logging_interval="step"),
]

trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    precision=16,
    gradient_clip_val=0.5,
    callbacks=callbacks,
    max_epochs=100,
)

if env.train:
    trainer.fit(model, data_module)
if env.test:
    trainer.test(model, data_module)
if env.restart:
    trainer.fit(model, data_module, ckpt_path=os.path.join(env["results"], "last.ckpt"))
