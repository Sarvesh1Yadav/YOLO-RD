import sys
from pathlib import Path
import os

import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    # 🔁 Resume checkpoint logic
    resume_ckpt = getattr(cfg.task, "resume_from_checkpoint", None)
    if resume_ckpt and not os.path.isfile(resume_ckpt):
        print(f"[⚠️] Checkpoint not found at {resume_ckpt}, starting fresh.")
        resume_ckpt = None

    # ✅ Add checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",   # Or change to "train_loss" or mAP depending on your metrics
        dirpath=os.path.join(save_path, "checkpoints"),
        filename="epoch={epoch}-step={step}",
        save_top_k=3,         # Keep only top 3 checkpoints
        every_n_epochs=1,
        save_last=True        # Save the last checkpoint too
    )
    callbacks.append(checkpoint_callback)

    # ⚙️ Trainer setup
    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="value",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quite", False),
        default_root_dir=save_path,
        **({"resume_from_checkpoint": resume_ckpt} if resume_ckpt else {})
    )

    # 🚀 Task selection
    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


if __name__ == "__main__":
    main()
