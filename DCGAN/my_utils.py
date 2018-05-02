import torch
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt



def imshow_with_tensor(img , cmap = None, block = True):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    print(img.size())
    plt.imshow(np.transpose(npimg, (1, 2,0)), cmap = cmap)
    plt.show(block = block)
    plt.pause(0.1)

def gradient_magnitude(inputs, channels = 1, device = 'cuda:0'):
    # grayscale
    g = torch.mean(inputs, dim=1).unsqueeze(1)
    # imshow(vutils.make_grid(g.detach()))

    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device))
    G_x = conv1(Variable(g))

    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).to(device))
    G_y = conv2(Variable(g))

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))

    G_result = G
    for i in range(channels-1) :
        G_result = torch.cat( (G_result,G), dim = 1)
    return G_result


'''

                filter = netD.conv1.weight.data.cpu().numpy()
                filter = np.transpose(filter, (0,2,3,1))
                # (1/(2*(maximum negative value)))*filter+0.5 === you need to normalize the filter before plotting.
                filter = (1 / (2 * 3.69201088)) * filter + 0.5  # Normalizing the values to [0,1]
'''
# num_cols= choose the grid size you want
def plot_kernels(tensor, num_cols=8):
    if not tensor.ndim == 4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1] == 3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

#plot_kernels(filter)