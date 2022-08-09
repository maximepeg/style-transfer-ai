from style.cnn import CNN
from yaml import safe_load as load_config
import torch

if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = load_config(file)
    test = CNN(config)

    sample = torch.ones((3, 100, 100))
    x = test(sample)
    print(x.shape)
