import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, config: dict):
        super(CNN, self).__init__()
        self.config = config
        self.feature_map_nb = config["nb_channel"]
        self.size = (self.config["size"], self.config["size"])
        self.first = nn.Conv2d(3, self.config['nb_channel'], kernel_size=(3, 3), padding='same')
        self.layer = self.init_layer(self.config['nb_channel'], self.config['nb_channel'])
        self.last = nn.Conv2d(self.config['nb_channel'], 3, kernel_size=(3, 3), padding='same')

    def init_layer(self, input_shape, out_shape, pooling=False):
        temp = []
        for i in range(self.config["nb_layer"]):
            pooling = i % 3 == 0
            temp.append(nn.Conv2d(input_shape, out_shape, padding="same", kernel_size=(3, 3)))
            temp.append(nn.SiLU())
        seq = nn.Sequential(*temp)
        return seq

    def forward(self, x):
        x = self.first(x)
        x = self.layer(x)
        x = self.last(x)

        return x