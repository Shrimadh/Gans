import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tensor_images(image_tensor, num_images=20, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    # plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    return image_grid

def get_noise(n_samples,noise_dim,device="cpu"):
    return  torch.randn(n_samples,noise_dim).to(device)


def get_gen_loss(gen,disc,criterion,n,z_dim,device):
    noise = get_noise(n,z_dim).to(device)
    fake = gen(noise)
    pred = disc(fake)
    loss = criterion(pred,torch.ones_like(pred))
    return loss

def get_disc_loss(gen,disc,crtierion,n,z_dim,real,device):
    noise = get_noise(n, z_dim).to(device)
    gen_image = gen(noise)
    gen_image = gen_image.detach()
    pred = disc(gen_image)
    loss_fake = criterion(pred,torch.zeros((n,1)).to(device))
    real_pred = disc(real)
    loss_real = criterion(real_pred,torch.ones((n,1)).to(device))
    disc_loss = (loss_fake+loss_real)/2.0
    return disc_loss


class Generator(nn.Module):
    def __init__(self,noise_dim=10,image_dim=1,hidden_dim=64):
        super(Generator,self).__init__()
        self.noise_dim = noise_dim
        self.gen = nn.Sequential(
            self.make_gen_block(noise_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, image_dim, kernel_size=4, final_layer=True),
        )
    def make_gen_block(self,input_channels,output_channels,kernel_size=3,stride=2,final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride),
                nn.Tanh()
            )
    def unsqueeze_noise(self,noise):
        return noise.view(len(noise),self.noise_dim,1,1)

    def forward(self,noise):
        noise = self.unsqueeze_noise(noise)
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self,image_channel = 1,hidden_dim = 16):
        super(Discriminator,self).__init__()
        self.dis = nn.Sequential(
            self.disc_block(image_channel, hidden_dim),
            self.disc_block(hidden_dim, hidden_dim * 2),
            self.disc_block(hidden_dim * 2, 1, final_layer=True),
        )
    def disc_block(self,input_channels,output_channels,kernel_size = 4,stride = 2,final_layer = False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,stride),
            )
    def forward(self,gen_image):
        pred = self.dis(gen_image)
        return pred.view(len(pred),-1)

# Loss function and some hyperparameters

criterion = nn.BCEWithLogitsLoss()
lr = 0.0002
n_epochs = 380
noise_dim = 64
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
dataloader = DataLoader(mnist,batch_size=batch_size,shuffle=True)

beta_1 = 0.5
beta_2 = 0.999

gen = Generator(noise_dim = noise_dim).to(device)
disc = Discriminator().to(device)
gen_optim = optim.Adam(gen.parameters(),lr = lr, betas=(beta_1, beta_2))
disc_optim = optim.Adam(disc.parameters(),lr = lr, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

writer_fake = SummaryWriter(f"DCGAN/MNIST/fake")
writer_real = SummaryWriter(f"DCGAN/MNIST/real")

gen_checkpoint = {'state_dict': gen.state_dict(),'optimizer' : gen_optim.state_dict()}
disc_checkpoint = {'state_dict': disc.state_dict(),'optimizer' : disc_optim.state_dict()}

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

def save_checkpoint(acc,checkpoint,path,mini):
    if acc < mini:
        mini = acc
        torch.save(checkpoint,path)

mini = 100     
#Training part

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False
step = 0
for epoch in range(n_epochs):
    for real,_ in tqdm(dataloader):
        b = len(real)

        disc.zero_grad()
        disc_loss = get_disc_loss(gen,disc,criterion,b,noise_dim,real,device)
        disc_loss.backward(retain_graph = True)
        disc_optim.step()

        gen_optim.zero_grad()
        gen_loss = get_gen_loss(gen,disc,criterion,b,noise_dim,device)
        gen_loss.backward(retain_graph=True)
        gen_optim.step()

        mean_discriminator_loss += disc_loss.item() / 500
        mean_generator_loss += gen_loss.item() / 500

        save_checkpoint(mean_generator_loss,gen_checkpoint,"DCGANgenerator.pth",mini)
        save_checkpoint(mean_generator_loss,disc_checkpoint,"DGGANdiscriminator.pth",mini)

        if cur_step % 500 == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(b, noise_dim, device=device)
            fake = gen(fake_noise)
            img_grid_fake = get_tensor_images(fake)
            img_grid_real = get_tensor_images(real)

            writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=step
            )
            writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=step
            )
            mean_generator_loss = 0
            mean_discriminator_loss = 0
            step += 1
        cur_step += 1
