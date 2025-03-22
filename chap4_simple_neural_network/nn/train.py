import yaml
from model import RegressionModel
from utils import RegressionDataset

def main():
    with open("../configs/net.yaml", 'r', encoding='utf-8') as file:
        config = yaml.load(file.read(), yaml.FullLoader) 
    
    model = RegressionModel(**config)
    dataset = RegressionDataset(model, train_percent=0.9)
    
    model.train(dataset)

if __name__ == "__main__":
    main()