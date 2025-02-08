def validation_check_intervals(config, len_train_loader):
    check_val_every_n_epoch = None
    val_check_interval = None
    if config.task.validation_interval > len_train_loader:
        check_val_every_n_epoch = config.task.check_val_every_n_epochs
    else:
        val_check_interval = config.task.validation_interval

    return val_check_interval, check_val_every_n_epoch