import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math


def lrp_individual(model, X, dataset, device="cpu"):
    mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1).to(device)
    std = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1).to(device)

    # Get the list of layers of the network
    layers = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)][1:]
    index_layer = len(layers[:-2])
    layers[-1:] = to_conv(layers[-1:], dataset)
    # Propagate the input
    L = len(layers)
    A = [X] + [None] * L # Create a list to store the activation produced by each layer
    for l in range(L): A[l+1] = layers[l].forward(A[l])

    # Get the relevance of the last layer using the highest classification score of the top layer
    T = A[-1].cpu().detach().numpy().tolist()[0]
    index = T.index(max(T))
    T = np.abs(np.array(T)) * 0
    T[index] = 1
    T = torch.FloatTensor(T)
    # Create the list of relevances with (L + 1) elements and assign the value of the last one
    R = [None] * L + [(A[-1].cpu() * T).data]

    # Propagation procedure from the top-layer towards the lower layers
    for l in range(1, L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l], torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(layers[l], torch.nn.AvgPool2d):

            if l <= 3:       rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z + 1e-9
            if 3 < l < 5: rho = lambda p: p;                       incr = lambda z: z + 1e-9 + 0.25 * (
                    (z ** 2).mean() ** .5).data
            if l >= 5:       rho = lambda p: p;                       incr = lambda z: z + 1e-9

            z = incr(newlayer(layers[l], rho).forward(A[l]))
            s = (R[l + 1].to(device) / z).data
            (z * s).sum().backward();
            c = A[l].grad
            R[l] = (A[l] * c).cpu().data
        else:
            R[l] = R[l + 1]

    if dataset == 'CIFAR10':
        lb = (A[0].data * 0 + (0 - mean) / std).requires_grad_(True)
        hb = (A[0].data * 0 + (1 - mean) / std).requires_grad_(True)
    else:
        lb = (A[0].data * 0 - 1).requires_grad_(True)
        hb = (A[0].data * 0 + 1).requires_grad_(True)

    A[0] = (A[0].data).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9
    z -= newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)
    z -= newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)
    s = (R[1].to(device) / z).data
    (z * s).sum().backward();
    c, cp, cm = A[0].grad, lb.grad, hb.grad
    R[0] = (A[0] * c + lb * cp + hb * cm).data
    orig_shape_r = R[0].size()
    R[0] = torch.FloatTensor([0 if math.isnan(x) else x for x in R[0].flatten()]).reshape(orig_shape_r)

    # Return the relevance of the input layer
    return np.array(R[0][0].cpu()).sum(axis=0)


def newlayer(layer,g):
    layer = copy.deepcopy(layer)
    try: layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError: pass
    try: layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer


def to_conv(layers, dataset):
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer, torch.nn.Linear):
            newlayer = None
            if i == 0:
                if dataset == 'FashionMNIST':
                    m,n = 32,layer.weight.shape[0]
                    newlayer = torch.nn.Conv2d(m,n,7)
                    newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n,m,7,7))
                else:
                    m, n = 32, layer.weight.shape[0]
                    newlayer = torch.nn.Conv2d(m, n, 8)
                    newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n, m, 8, 8))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = torch.nn.Conv2d(m,n,1)
                newlayer.weight = torch.nn.Parameter(layer.weight.reshape(n,m,1,1))
            newlayer.bias = torch.nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers


def heatmap(R,sx,sy, img):
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    plt.imshow(img, cmap='gray', alpha=0.2)
    plt.show()


def digit(X,sx,sy):
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(X,interpolation='nearest',cmap='gray')
    plt.show()