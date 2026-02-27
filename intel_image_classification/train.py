from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from utils import figure_of_confusion_matrix


class Trainer:
    """
    Attributes:
        checkpoint_path (str): Path to the folder with checkpoins ("checkpoints_path/").
    """

    def __init__(self, model, loaders, optimization, log, config):
        self.model = model

        self.train_loader = loaders["train_loader"]
        self.valid_loader = loaders["valid_loader"]

        self.criterion = optimization["criterion"]
        self.optimizer = optimization["optimizer"]
        self.scheduler = optimization["scheduler"]

        self.metrics = log["metrics"]
        self.conf_matrix = log["conf_matrix"]
        self.writer = log["writer"]

        self.config = config

        self.best_f1 = torch.tensor([0.0])
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        train_pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )
        for i, (batch, labels) in train_pbar:
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.writer.add_scalar("loss/train", loss.item(), self.global_step)

            # Логируем только lr для head (последние слои, самый большой lr)
            self.writer.add_scalar(
                "lr/train", self.optimizer.param_groups[-1]["lr"], self.global_step
            )
            self.global_step += 1

    def validate(self, epoch):
        self.model.eval()
        val_loss = torch.tensor([0.0])
        self.metrics.reset()
        self.conf_matrix.reset()
        val_pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        with torch.no_grad():
            for batch, labels in val_pbar:
                out = self.model(batch)
                val_loss += self.criterion(out, labels)
                preds = out.argmax(dim=1)
                self.metrics.update(preds, labels)
                self.conf_matrix.update(preds, labels)

        self.writer.add_scalar(
            "loss/validation", val_loss / len(self.valid_loader), epoch
        )
        metrics_res = self.metrics.compute()
        for key, value in metrics_res.items():
            self.writer.add_scalar(f"{key}/validation", value, epoch)
        fig = figure_of_confusion_matrix(
            self.conf_matrix.compute(), self.config["classes"]
        )
        self.writer.add_figure("confusion_matrix", fig, epoch)
        plt.close(fig)

        save_dir = Path(self.config["checkpoint_path"])
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "f1": self.best_f1,
        }
        torch.save(checkpoint, save_dir / "last_model_checkpoint.pth")

        if metrics_res["F1"] > self.best_f1:
            self.best_f1 = metrics_res["F1"]
            torch.save(
                checkpoint, self.config["checkpoint_path"] + "best_model_checkpoint.pth"
            )
            tqdm.write(f"Best F1: {self.best_f1:.4f} saved at epoch {epoch}")

    def fit(self, num_epochs, start_epoch=0):
        epoch_pbar = tqdm(
            range(start_epoch, num_epochs), desc="Total Progress", leave=True
        )
        try:
            for epoch in epoch_pbar:
                self.train_epoch(epoch)
                self.validate(epoch)
        finally:
            self.writer.close()

    def load_checkpoint(self, path="checkpoints/best_model_checkpoint.pth"):
        """Load checkpoint

        Args:
            path (str, optional): Path to checkpoint. Defaults to \"checkpoints/best_model_checkpoint.pth\".
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_f1 = checkpoint["f1"]
        self.global_step = checkpoint["global_step"]
        last_epoch = checkpoint["epoch"]
        tqdm.write(f"Resuming from epoch {last_epoch + 1}")
        return last_epoch + 1

    def test(self):
        return Path(self.config["checkpoint_path"])


if __name__ == "__main__":
    pass
