import time
import numpy as np
import matplotlib.pyplot as plt
import nn

class Dataset():
    def __init__(self, x, y, train_percent: float):
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.ndim == 2 and y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        split = int(train_percent * self.x.shape[0])
        self.x_train = self.x[:split]
        self.y_train = self.y[:split]
        self.x_test = self.x[split:]
        self.y_test = self.y[split:]
    
    def __len__(self, ):
        return self.x_train.shape[0]
    
    def iterate_once(self, batch_size):
        assert isinstance(batch_size, int)
        data_num = len(self) // batch_size
        if len(self) % batch_size != 0:
            data_num += 1 # add truncated batch
        i = 0
        while i < data_num:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self))
            x = self.x_train[start_idx:end_idx]
            y = self.y_train[start_idx:end_idx]
            yield nn.Constant(x), nn.Constant(y)
            i += 1
    
    def get_validation_loss(self, ):
        raise NotImplementedError

class RegressionDataset(Dataset):
    def __init__(self, model, train_percent = 0.9):
        self.x_lim = [-2 * np.pi, 2 * np.pi]
        x = np.linspace(*self.x_lim, num=300)[:, None] # shape (data_num, 1)
        np.random.RandomState(0).shuffle(x)
        self.argsort_x = np.argsort(x.flatten())
        y = np.sin(x)
        super().__init__(x, y, train_percent)
        
        self.model = model
        self.processed = 0
        
        # visualization for regression
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_xlim(*self.x_lim)
        ax.set_ylim(-1.4, 1.4)
        real, = ax.plot(x[self.argsort_x], y[self.argsort_x], color='blue')
        learned, = ax.plot([], [], color='red')
        text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va='top')
        ax.legend([real, learned], ["true", "predicted"])
        plt.grid(True)
        plt.show(block=False)
        
        self.fig = fig
        self.learned = learned
        self.text = text
        self.last_update = time.time()
    
    def iterate_once(self, batch_size):
        for x, y in super().iterate_once(batch_size):
            yield x, y
            self.processed += batch_size
        
        # 10Hz update rate for visualization
        if time.time() - self.last_update > 0.1:
            predicted = self.model(nn.Constant(self.x)).data
            loss = self.model.criterion()(nn.Constant(predicted), nn.Constant(self.y)).data
            self.learned.set_data(self.x[self.argsort_x], predicted[self.argsort_x])
            self.text.set_text(f"processed: {self.processed}\nloss: {loss:.3f}")
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(1e-3)
            self.last_update = time.time()
    
    def get_validation_loss(self):
        y_pred = self.model(nn.Constant(self.x_test))
        loss = self.model.criterion()(y_pred, nn.Constant(self.y_test)).data
        return loss