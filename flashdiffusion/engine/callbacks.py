"""Training callbacks / hooks system.

Extensible hook points for the training loop. Users can add custom
behavior without modifying the Trainer source code.

Usage:
    from flashdiffusion.engine.callbacks import Callback, CallbackList

    class WandbLogger(Callback):
        def on_train_end(self, trainer, metrics):
            wandb.log(metrics)

    trainer = Trainer(...)
    trainer.add_callback(WandbLogger())
    trainer.train()
"""

from typing import Any, Dict, List, Optional


class Callback:
    """Base class for training callbacks.

    Override any method to hook into the training loop.
    All methods receive the trainer instance as the first argument.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called once before training begins."""

    def on_train_end(self, trainer: Any, metrics: Dict) -> None:
        """Called once after training completes."""

    def on_step_start(self, trainer: Any, step: int) -> None:
        """Called at the start of each training step."""

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        """Called at the end of each training step."""

    def on_checkpoint(self, trainer: Any, path: str) -> None:
        """Called when a checkpoint is saved."""

    def on_validation(self, trainer: Any, metrics: Dict) -> None:
        """Called after validation."""


class CallbackList:
    """Manages a list of callbacks, dispatching events to all of them."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks: List[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def fire(self, event: str, *args, **kwargs) -> None:
        """Fire an event on all registered callbacks."""
        for cb in self.callbacks:
            method = getattr(cb, event, None)
            if method:
                method(*args, **kwargs)


class EarlyStopping(Callback):
    """Stop training when loss stops improving."""

    def __init__(self, patience: int = 500, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.should_stop = False

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                print(f"EarlyStopping: loss did not improve for {self.patience} steps. Stopping.")


class CSVLogger(Callback):
    """Log training metrics to a CSV file."""

    def __init__(self, path: str = "training_log.csv"):
        self.path = path
        self._initialized = False

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        import csv

        row = {"step": step, "loss": loss}
        if not self._initialized:
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            self._initialized = True
        else:
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self._writer = None

    def on_train_start(self, trainer: Any) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(self.log_dir)

    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if self._writer:
            self._writer.add_scalar("train/loss", loss, step)

    def on_train_end(self, trainer: Any, metrics: Dict) -> None:
        if self._writer:
            self._writer.close()
