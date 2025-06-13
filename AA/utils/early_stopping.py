class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and model is not None:
                # 深度复制模型状态
                import copy
                self.best_weights = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_best_weights_to_model(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)