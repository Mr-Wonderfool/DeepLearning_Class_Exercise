import nn

class RegressionModel():
    def __init__(self, **kwargs):
        configs = kwargs[str(self)]
        hidden_dims = configs.pop('mlp_dims')
        input_dims = hidden_dims.pop(0)
        output_dims = hidden_dims.pop(-1)
        assert len(hidden_dims) == 2
        self.w1 = nn.Parameter(input_dims, hidden_dims[0])
        self.b1 = nn.Parameter(1, hidden_dims[0])
        self.w2 = nn.Parameter(hidden_dims[0], hidden_dims[1])
        self.b2 = nn.Parameter(1, hidden_dims[1])
        self.w3 = nn.Parameter(hidden_dims[1], output_dims)
        self.b3 = nn.Parameter(1, output_dims)
        
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        
        self.batch_size = configs.pop('batch_size')
        self.lr = configs.pop('lr')
        self.tolerance = configs.pop('tolerance')
    
    def __call__(self, x):
        z1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        a1 = nn.ReLU(z1)
        z2 = nn.AddBias(nn.Linear(a1, self.w2), self.b2)
        a2 = nn.ReLU(z2)
        z3 = nn.AddBias(nn.Linear(a2, self.w3), self.b3)
        return z3
    
    def criterion(self):
        return nn.SquareLoss
    
    def train(self, dataset):
        """
        :param tolerance: threshold for loss on test set
        """
        total_loss = self.tolerance
        while total_loss >= self.tolerance:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.criterion()(self(x), y)
                grads = nn.backprop(loss, self.params)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)
            total_loss = dataset.get_validation_loss()
    
    def __str__(self):
        return "RegressionModel"