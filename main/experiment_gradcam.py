import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy import stats
import random
import os
import csv
from CustomDataset import CustomDataset
from GradCAMModel import *
import math
import sys


dataset = sys.argv[1]
model_n = int(float(sys.argv[2]))
epochs = int(float(sys.argv[3]))
RANDOM_SEED = int(float(sys.argv[4]))
# random seeds
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def add_item_to_csv(file_name, dataset, item):
    newpath = './csvs_gradcam/{}/{}_{}.csv'.format(dataset, file_name, model_n)
    mode = 'a'
    if not os.path.exists(newpath):
        mode = 'w'
    with open(newpath, mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow([item])


def compute_gradcam_values(model, images):
    inputs = images.to(device)
    logits, predicted = torch.max(model(inputs).data, 1)

    negative_array, positive_array, pos_n_array, neg_n_array, \
    norm_array, pos_entropy_array, neg_entropy_array = [], [], [], [], [], [], []
    heatmaps = []

    for index, logit in enumerate(logits):
        logit.requires_grad_()
        logit.backward()

        # pull the gradients out of the model
        gradients = model.get_activations_gradient().detach().cpu()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(inputs[index].reshape(1, inputs[index].shape[0], inputs[index].shape[1], inputs[index].shape[1])).detach().cpu()

        # weight the channels by corresponding gradients
        for k in range(activations.shape[1]):
            activations[:, k, :, :] *= pooled_gradients[k]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmaps.append(heatmap)
        norm_array.append(np.linalg.norm(heatmap.flatten()))

        heatmap_positive = np.maximum(heatmap.numpy(), 0)
        heatmap_negative = np.minimum(heatmap.numpy(), 0)
        # relu on top of the heatmap
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)

        negative_array.append(sum(heatmap_negative.flatten() < 0) / len(heatmap_negative.flatten()))
        positive_array.append(sum(heatmap_positive.flatten() > 0) / len(heatmap_positive.flatten()))
        pos_n_array.append(np.linalg.norm(heatmap_positive.flatten()))
        neg_n_array.append(np.linalg.norm(heatmap_negative.flatten()))
        pos_entropy_array.append(stats.entropy(heatmap_positive.flatten()))
        neg_entropy_array.append(stats.entropy(heatmap_negative.flatten()))

    neg_entropy_array = [0 if math.isnan(x) else x for x in neg_entropy_array]
    pos_entropy_array = [0 if math.isnan(x) else x for x in pos_entropy_array]
    add_item_to_csv('negative_gradcam', dataset, negative_array)
    add_item_to_csv('positive_gradcam', dataset, positive_array)
    add_item_to_csv('positive_gradcam_norm', dataset, pos_n_array)
    add_item_to_csv('negative_gradcam_norm', dataset, neg_n_array)
    add_item_to_csv('gradcam_norm', dataset, norm_array)
    add_item_to_csv('positive_gradcam_entropy', dataset, pos_entropy_array)
    add_item_to_csv('negative_gradcam_entropy', dataset, neg_entropy_array)
    return torch.stack(heatmaps)


def train():
    if dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), )
                                        ])
        train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                                      transform=transform,
                                                      download=True)
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False,
                                                     transform=transform,
                                                     download=True)
        model = GradCAMModelCIFAR10()
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,), )
                                        ])
        train_set = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                      transform=transform,
                                                      download=True)
        test_set = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                     transform=transform,
                                                     download=True)
        model = GradCAMModel()

    images = {}
    images['label'] = []
    images['label_int'] = []
    images['images'] = []
    for label, value in train_set.class_to_idx.items():
        images['label'].extend([label]*20)
        images['label_int'].extend([value]*20)
        images['images'].extend([img for index, img in enumerate(train_set.data)
                                   if train_set.targets[index] == value][:10])
        images['images'].extend([img for index, img in enumerate(test_set.data)
                                   if test_set.targets[index] == value][:10])
    if dataset == 'CIFAR10':
        images['images'] = [torch.from_numpy(img) for img in images['images']]

    train_dataloader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=100, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    criterion_all = nn.CrossEntropyLoss(reduction='none')

    gradcam_step_size = 1

    gradcam_image_dataset = CustomDataset(images, transform)
    gradcam_image_dataset.transform = transform
    gradcam_dataloader = DataLoader(gradcam_image_dataset, batch_size=len(gradcam_image_dataset))

    newpath = './results_gradcam/{}'.format(dataset)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = './csvs_gradcam/{}'.format(dataset)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = './gradcam/{}'.format(dataset)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    gradcam_per_class = {classname: [] for classname in images['label_int']}
    accuracy_per_class = {classname: [] for classname in images['label_int']}
    total_step = len(train_dataloader)
    heatmaps_array = []

    model.to(device)

    for epoch in range(epochs):

        epoch_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            correct_pred_train = {classname: 0 for classname in train_set.classes}
            total_pred_train = {classname: 0 for classname in train_set.classes}
            loss_per_class = {classname: 0 for classname in train_set.classes}

            model.train()
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_all = criterion_all(outputs, labels).detach().cpu().numpy()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += outputs.shape[0] * loss.item()

            add_item_to_csv('train_loss', dataset, loss.item())
            add_item_to_csv('train_accuracy', dataset, 100*correct / total)

            for label, train_loss in zip(labels, loss_all):
                loss_per_class[train_set.classes[label]] += train_loss
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred_train[train_set.classes[label]] += 1
                total_pred_train[train_set.classes[label]] += 1

            for classname, correct_count in correct_pred_train.items():
                if total_pred_train[classname] != 0:
                    accuracy = float(correct_count) / total_pred_train[classname]
                else:
                    accuracy = 0
                add_item_to_csv('train_accuracy_class_{}'.format(train_set.class_to_idx[classname]), dataset, accuracy)

            for classname, train_loss in loss_per_class.items():
                if total_pred_train[classname] != 0:
                    train_loss_class = float(train_loss) / total_pred_train[classname]
                else:
                    train_loss_class = 0
                add_item_to_csv('train_loss_class_{}'.format(train_set.class_to_idx[classname]), dataset,
                                train_loss_class)

            if (i+1) % gradcam_step_size == 0:
                # print statistics
                print('Epoch [{}/{}], Step [{}/{}], Loss: {}, Running Loss: {}'.format(
                    epoch, epoch, i + 1, total_step, loss.item(), running_loss/(i+1)))

                add_item_to_csv('running_loss', dataset, running_loss/(i+1))

                # gradcam values
                model.eval()
                for data_gradcam in gradcam_dataloader:
                    inputs_gradcam, labels_gradcam = data_gradcam
                    heatmaps = compute_gradcam_values(model, inputs_gradcam.detach().clone())
                    heatmaps_array.append(heatmaps)

                # Testing Loss
                correct = 0
                total = 0
                testing_loss = 0.0

                # prepare to count predictions for each class
                correct_pred = {classname: 0 for classname in test_set.classes}
                total_pred = {classname: 0 for classname in test_set.classes}
                test_loss_per_class = {classname: 0 for classname in test_set.classes}

                with torch.no_grad():
                    for data in test_dataloader:
                        inputs, labels = data[0].to(device), data[1].to(device)
                        # calculate outputs by running images through the network
                        outputs = model(inputs)
                        testing_loss += criterion(outputs, labels).item()
                        loss_all = criterion_all(outputs, labels).detach().cpu().numpy()
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        # collect the correct predictions for each class
                        for label, prediction in zip(labels, predicted):
                            if label == prediction:
                                correct_pred[test_set.classes[label]] += 1
                            total_pred[test_set.classes[label]] += 1
                        for label, loss in zip(labels, loss_all):
                            test_loss_per_class[test_set.classes[label]] += loss

                add_item_to_csv('test_loss', dataset, testing_loss/len(test_dataloader))

                correct_gradcam = 0
                total_gradcam = 0
                testing_loss_gradcam = 0.0

                correct_pred_gradcam = {classname: 0 for classname in test_set.classes}
                total_pred_gradcam = {classname: 0 for classname in test_set.classes}

                with torch.no_grad():
                    for data_gradcam in gradcam_dataloader:
                        inputs_gradcam, labels_gradcam = data_gradcam[0].to(device), data_gradcam[1].to(device)
                        # calculate outputs by running images through the network
                        outputs_gradcam = model(inputs_gradcam.float())
                        testing_loss_gradcam += criterion(outputs_gradcam, labels_gradcam).item()
                        add_item_to_csv('test_gradcam_loss', dataset, criterion(outputs_gradcam, labels_gradcam).item())
                        loss_all = criterion_all(outputs_gradcam, labels_gradcam).detach().cpu().numpy()
                        add_item_to_csv('test_gradcam_loss_all', dataset, loss_all)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted_gradcam = torch.max(outputs_gradcam.data, 1)
                        total_gradcam += labels_gradcam.size(0)
                        correct_gradcam += (predicted_gradcam == labels_gradcam).sum().item()

                        for label, prediction in zip(labels_gradcam, predicted_gradcam):
                            if label == prediction:
                                correct_pred_gradcam[test_set.classes[label]] += 1
                            total_pred_gradcam[test_set.classes[label]] += 1

                add_item_to_csv('test_accuracy', dataset, 100*correct / total)

                add_item_to_csv('test_gradcam_accuracy', dataset, 100*correct_gradcam / total_gradcam)

                for classname, correct_count in correct_pred.items():
                    accuracy = float(correct_count) / total_pred[classname]
                    accuracy_per_class[test_set.class_to_idx[classname]].append(accuracy)
                    add_item_to_csv('test_accuracy_class_{}'.format(test_set.class_to_idx[classname]), dataset, accuracy)

                for classname, correct_count in correct_pred_gradcam.items():
                    accuracy = float(correct_count) / total_pred_gradcam[classname]
                    gradcam_per_class[test_set.class_to_idx[classname]].append(accuracy)
                    add_item_to_csv('test_accuracy_gradcam_class_{}'.format(test_set.class_to_idx[classname]), dataset, accuracy)

                for classname, loss in test_loss_per_class.items():
                    loss = float(loss) / total_pred[classname]
                    add_item_to_csv('test_loss_class_{}'.format(test_set.class_to_idx[classname]), dataset, loss)

    print('Finished Training')
    if model_n == 3:
        torch.save(torch.FloatTensor(torch.stack(heatmaps_array)), './gradcam/{}/gradcam_values_model_{}'.format(dataset, model_n))
        torch.save(model.state_dict(), "./gradcam/model_{}_{}.ckpt".format(model_n, dataset))


if __name__ == "__main__":
    train()
