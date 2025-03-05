def validation_check_intervals(config, len_train_loader):
    check_val_every_n_epoch = None
    val_check_interval = None
    if config.task.validation_interval > len_train_loader:
        check_val_every_n_epoch = config.task.check_val_every_n_epochs
    else:
        val_check_interval = config.task.validation_interval

    return val_check_interval, check_val_every_n_epoch

import pytorch_lightning as pl

class StepBasedEarlyStopping(pl.Callback):
    def __init__(self, monitor="val_auc", mode="max", patience=1000, min_delta=1e-5):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.no_improve_steps = 0
        self.last_step = 0  # Ensure last_step is initialized

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            return  # Skip if metric isn't available

        # Check if improvement happened
        if self.best_score is None or (
            self.mode == "max" and current_score > self.best_score + self.min_delta
        ) or (self.mode == "min" and current_score < self.best_score - self.min_delta):
            self.best_score = current_score
            self.no_improve_steps = 0  # Reset counter
        else:
            # Accumulate steps since last improvement
            self.no_improve_steps += trainer.global_step - self.last_step

        self.last_step = trainer.global_step  # Update last tracked step

        # Stop training if patience threshold is reached
        if self.no_improve_steps >= self.patience:
            print(f'No improvement in {self.monitor} for {self.no_improve_steps} steps')
            trainer.should_stop = True

