"""
Week 4 Training Script
Cross-Modal SAR–Multispectral Satellite Image Retrieval

Training setup:
- PyTorch Lightning
- Two-tower ResNet-50 model
- Symmetric InfoNCE loss
- Physical batch size: 16
- Mixed precision: 16-mixed
- Effective training dataset: 21,000 pairs
- Maximum epochs: 20
- Automatic GPU selection
- Gradient clipping: 1.0
- Best checkpoint selected using validation loss
- CSV training logs
- Training-loss curve
"""

from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataloader import create_paired_dataloaders
from src.models.two_tower import InfoNCELoss, TwoTowerNetwork


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = Path("outputs/metadata/train_split.csv")
VAL_CSV = Path("outputs/metadata/val_split.csv")

S1_ROOT = Path(
    r"C:\BigEarthNet-S1-Required\BigEarthNet-S1-Required"
)

S2_ROOT = Path(
    r"C:\BigEarthNet-S2-Required"
)

CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_DIR = Path("outputs/logs")
PLOT_DIR = Path("outputs/plots")


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 4

MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
TEMPERATURE = 0.07

GRADIENT_CLIP_VAL = 1.0

# Use mixed precision to reduce VRAM usage.
PRECISION = "16-mixed"


# ============================================================
# LIGHTNING MODULE
# ============================================================

class RetrievalLightningModule(L.LightningModule):
    """
    Lightning wrapper around the two-tower retrieval network.
    """

    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        temperature: float = TEMPERATURE,
        max_epochs: int = MAX_EPOCHS,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.network = TwoTowerNetwork(pretrained=True)

        self.loss_fn = InfoNCELoss(
            temperature=temperature
        )

    def forward(
        self,
        sar_batch: torch.Tensor,
        ms_batch: torch.Tensor,
    ):
        return self.network(
            sar_batch,
            ms_batch,
        )

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):
        sar_batch, ms_batch, _, _ = batch

        sar_embeddings, ms_embeddings = self(
            sar_batch,
            ms_batch,
        )

        loss = self.loss_fn(
            sar_embeddings,
            ms_embeddings,
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=sar_batch.size(0),
        )

        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        sar_batch, ms_batch, _, _ = batch

        sar_embeddings, ms_embeddings = self(
            sar_batch,
            ms_batch,
        )

        loss = self.loss_fn(
            sar_embeddings,
            ms_embeddings,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=sar_batch.size(0),
        )

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ============================================================
# TRAINING CURVE
# ============================================================

def create_training_plot(
    metrics_path: Path,
) -> None:
    """
    Create train/validation loss plot from Lightning CSV logs.
    """

    if not metrics_path.exists():
        print(
            f"[WARNING] Metrics file not found: {metrics_path}"
        )
        return

    metrics = pd.read_csv(metrics_path)

    train_metrics = metrics[
        ["epoch", "train_loss"]
    ].dropna()

    val_metrics = metrics[
        ["epoch", "val_loss"]
    ].dropna()

    if train_metrics.empty and val_metrics.empty:
        print("[WARNING] No loss data found for plotting.")
        return

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 6))

    if not train_metrics.empty:
        plt.plot(
            train_metrics["epoch"],
            train_metrics["train_loss"],
            marker="o",
            label="Train Loss",
        )

    if not val_metrics.empty:
        plt.plot(
            val_metrics["epoch"],
            val_metrics["val_loss"],
            marker="o",
            label="Validation Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.title(
        "Cross-Modal Retrieval Training Loss"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = (
        PLOT_DIR / "training_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"[OK] Training plot saved: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # Tensor Core optimization for NVIDIA GPU.
    torch.set_float32_matmul_precision("high")

    print("=" * 70)
    print(
        "WEEK 4 — CROSS-MODAL RETRIEVAL TRAINING"
    )
    print("=" * 70)
    print()

    print("Configuration:")
    print(f"Train CSV          : {TRAIN_CSV}")
    print(f"Validation CSV     : {VAL_CSV}")
    print(f"S1 Root            : {S1_ROOT}")
    print(f"S2 Root            : {S2_ROOT}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Workers            : {NUM_WORKERS}")
    print(f"Maximum epochs     : {MAX_EPOCHS}")
    print(f"Learning rate      : {LEARNING_RATE}")
    print(f"Temperature        : {TEMPERATURE}")
    print(
        f"Gradient clipping  : "
        f"{GRADIENT_CLIP_VAL}"
    )
    print(
        f"Precision          : "
        f"{PRECISION}"
    )
    print()

    # --------------------------------------------------------
    # CHECK REQUIRED PATHS
    # --------------------------------------------------------

    required_paths = {
        "Train CSV": TRAIN_CSV,
        "Validation CSV": VAL_CSV,
        "S1 root": S1_ROOT,
        "S2 root": S2_ROOT,
    }

    for name, path in required_paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

        print(
            f"[OK] {name}: {path}"
        )

    print()

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CREATE DATALOADERS
    # --------------------------------------------------------

    print(
        "Creating training and validation "
        "DataLoaders..."
    )
    print()

    train_loader, val_loader = (
        create_paired_dataloaders(
            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            s1_root=S1_ROOT,
            s2_root=S2_ROOT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
    )

    print()

    print(
        f"[OK] Training samples   : "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"[OK] Validation samples : "
        f"{len(val_loader.dataset):,}"
    )

    print()

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print(
        "Creating two-tower model..."
    )

    model = RetrievalLightningModule(
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
        max_epochs=MAX_EPOCHS,
    )

    print("[OK] Model created.")
    print()

    # --------------------------------------------------------
    # CHECKPOINT CALLBACK
    # --------------------------------------------------------

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # --------------------------------------------------------
    # CSV LOGGER
    # --------------------------------------------------------

    csv_logger = CSVLogger(
        save_dir=LOG_DIR,
        name="retrieval_training",
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        precision=PRECISION,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        log_every_n_steps=1,
    )

    # --------------------------------------------------------
    # START TRAINING
    # --------------------------------------------------------

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    start_time = time.perf_counter()

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    elapsed_hours = (
        elapsed_seconds / 3600.0
    )

    # --------------------------------------------------------
    # READ TRAINING METRICS
    # --------------------------------------------------------

    metrics_path = (
        Path(csv_logger.log_dir)
        / "metrics.csv"
    )

    final_train_loss = None
    final_val_loss = None
    best_val_loss = None
    best_epoch = None

    if metrics_path.exists():

        metrics = pd.read_csv(
            metrics_path
        )

        train_metrics = metrics[
            ["epoch", "train_loss"]
        ].dropna()

        val_metrics = metrics[
            ["epoch", "val_loss"]
        ].dropna()

        if not train_metrics.empty:

            final_train_loss = float(
                train_metrics.iloc[-1][
                    "train_loss"
                ]
            )

        if not val_metrics.empty:

            final_val_loss = float(
                val_metrics.iloc[-1][
                    "val_loss"
                ]
            )

            best_index = val_metrics[
                "val_loss"
            ].idxmin()

            best_val_loss = float(
                val_metrics.loc[
                    best_index,
                    "val_loss",
                ]
            )

            best_epoch = int(
                val_metrics.loc[
                    best_index,
                    "epoch",
                ]
            )

    # --------------------------------------------------------
    # CREATE TRAINING PLOT
    # --------------------------------------------------------

    create_training_plot(
        metrics_path
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Total training time : "
        f"{elapsed_hours:.4f} hours"
    )

    print(
        f"Final training loss : "
        f"{final_train_loss}"
    )

    print(
        f"Final validation loss: "
        f"{final_val_loss}"
    )

    print(
        f"Best validation loss : "
        f"{best_val_loss}"
    )

    print(
        f"Best validation epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best checkpoint      : "
        f"{checkpoint_callback.best_model_path}"
    )

    print(
        f"Last checkpoint      : "
        f"{checkpoint_callback.last_model_path}"
    )

    print(
        f"Training logs        : "
        f"{csv_logger.log_dir}"
    )

    print(
        f"Training plot        : "
        f"{PLOT_DIR / 'training_loss.png'}"
    )

    print()

    print("=" * 70)
    print(
        "WEEK 4 TRAINING RUN FINISHED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()