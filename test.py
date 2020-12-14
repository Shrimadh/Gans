import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def gen_block(dim1,dim2):
    return nn.Sequential(
        nn.Linear(dim1,dim2),
        nn.BatchNorm1d(dim2),
        nn.ReLU(inplace=True)
    )

def get_noise(n_samples,noise_dim,device="cpu"):
    return  torch.randn(n_samples,noise_dim).to(device)

class Generator(nn.Module):
    def __init__(self,noise_dim=10,image_dim=784,hidden_dim=128):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            gen_block(noise_dim,hidden_dim),
            gen_block(hidden_dim,hidden_dim*2),
            gen_block(hidden_dim*2,hidden_dim*4),
            gen_block(hidden_dim*4,hidden_dim*8),
            nn.Linear(hidden_dim*8,image_dim),
            nn.Sigmoid()
        )
    def forward(self,noise):
        return self.gen(noise)
gen = Generator(64)
gen.load_state_dict(torch.load("generator.pth",map_location = "cpu"))
gen.eval()

fake = gen(get_noise(128,64))
show_tensor_images(fake,1)