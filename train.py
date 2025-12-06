import torch
from Agent import Agent
from QNetwork import *
from Environment import *

if __name__ == '__main__':
    device = torch.device("cuda")
    e = DyckNTaskEnvironment(device)
    net = QNetwork(9, 7, 10, device, q_layer_num=2)
    net.load_model('model/02-3.pth')
    a = Agent(e, net, 9, 7, 10, 10000, device=device)
    a.train(10000, 256, 5,  4)
    a.target_network.save_model('model/02-4.pth')
