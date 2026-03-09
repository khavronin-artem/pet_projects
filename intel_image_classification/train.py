from pathlib import Path

import matplotlib.pyplot as plt
import torch
import utils
from tqdm.auto import tqdm
from utils import figure_of_confusion_matrix


class Trainer:
    """
    Attributes:
        checkpoint_path (str): Path to the folder with checkpoins ("checkpoints_path/").
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data: utils.DataAndLoaders,
        opt: utils.OptimizationSuite,
        log: utils.LoggingSuite,
        conf: utils.TrainerConfig,
    ):
        self.model = model
        self.data = data
        self.opt = opt
        self.log = log
        self.conf = conf

        self.best_f1 = torch.tensor([0.0])
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        train_pbar = tqdm(
            enumerate(self.data.train_loader),
            total=len(self.data.train_loader),
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )
        for i, (batch, labels) in train_pbar:
            self.opt.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.opt.criterion(out, labels)
            loss.backward()
            self.opt.optimizer.step()
            self.opt.scheduler.step()

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.log.writer.add_scalar("loss/train", loss.item(), self.global_step)

            # Логируем только lr для head (последние слои, самый большой lr)
            # ПОДУМАТЬ НАД ЭТИМ!!!
            self.log.writer.add_scalar(
                "lr/train", self.opt.optimizer.param_groups[-1]["lr"], self.global_step
            )
            self.global_step += 1

    def validate(self, epoch):
        self.model.eval()
        val_loss = torch.tensor([0.0])
        self.log.metrics.reset()
        self.log.conf_matrix.reset()
        val_pbar = tqdm(
            self.data.valid_loader, desc=f"Epoch {epoch} [Val]", leave=False
        )
        with torch.no_grad():
            for batch, labels in val_pbar:
                out = self.model(batch)
                val_loss += self.opt.criterion(out, labels)
                preds = out.argmax(dim=1)
                self.log.metrics.update(preds, labels)
                self.log.conf_matrix.update(preds, labels)

        self.log.writer.add_scalar(
            "loss/validation", val_loss / len(self.data.valid_loader), epoch
        )
        metrics_res = self.log.metrics.compute()
        for key, value in metrics_res.items():
            self.log.writer.add_scalar(f"{key}/validation", value, epoch)
        fig = figure_of_confusion_matrix(
            self.log.conf_matrix.compute(), self.data.classes
        )
        self.log.writer.add_figure("confusion_matrix", fig, epoch)
        plt.close(fig)

        save_dir = Path(self.conf.checkpoint_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.optimizer.state_dict(),
            "scheduler_state_dict": self.opt.scheduler.state_dict(),
            "f1": self.best_f1,
        }
        torch.save(checkpoint, save_dir / "last_model_checkpoint.pth")

        if metrics_res["F1"] > self.best_f1:
            self.best_f1 = metrics_res["F1"]
            torch.save(checkpoint, save_dir / "best_model_checkpoint.pth")
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
            self.log.writer.close()

    def load_checkpoint(
        self, path="checkpoints/best_model_checkpoint.pth", verbose=True
    ):
        """Load checkpoint

        Args:
            path (str, optional): Path to checkpoint. Defaults to \"checkpoints/best_model_checkpoint.pth\".
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.opt.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_f1 = checkpoint["f1"]
        self.global_step = checkpoint["global_step"]
        last_epoch = checkpoint["epoch"]
        if verbose:
            tqdm.write(f"Resuming from epoch {last_epoch + 1}")
        return last_epoch + 1


if __name__ == "__main__":
    pass
