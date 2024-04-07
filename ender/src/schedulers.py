class LearningRateSchedulerBase:
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr

    def schedule(self, epoch):
        raise NotImplementedError("Subclasses must implement the schedule method.")

class LearningRateScheduler(LearningRateSchedulerBase):
    def __init__(self, initial_lr, factor, epochs_drop):
        super().__init__(initial_lr)
        self.factor = factor
        self.epochs_drop = epochs_drop

    def schedule(self, epoch):
        lr = self.initial_lr * (self.factor ** (epoch // self.epochs_drop))
        return lr
    
class ExponentialDecayScheduler(LearningRateSchedulerBase):
    def __init__(self, initial_lr, decay_rate):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def schedule(self, epoch):
        lr = self.initial_lr * (self.decay_rate ** epoch)
        return lr

class ReduceLROnPlateauScheduler(LearningRateSchedulerBase):
    def __init__(self, initial_lr, factor, patience):
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0

    def schedule(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                lr = self.initial_lr * self.factor
                self.initial_lr= self.initial_lr * self.factor
                self.wait = 0
                return lr
        return None

class OneCycleLR(LearningRateSchedulerBase):
    def __init__(self, max_lr, epochs, cycle_fraction=0.3):
        super().__init__(max_lr)
        self.epochs = epochs
        self.cycle_fraction = cycle_fraction

    def schedule(self, epoch):
        cycle_epochs = int(self.epochs * self.cycle_fraction)
        if epoch < cycle_epochs:
            lr = self.initial_lr * (epoch / cycle_epochs)
        else:
            lr = self.initial_lr * (1 - (epoch - cycle_epochs) / (self.epochs - cycle_epochs))
        return lr
