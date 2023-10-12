import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import os
import torchvision.transforms as transforms
import random
from scipy import stats
from CustomDataset import CustomDataset
from LRPModel import *
import torch
import csv
from LRPValues import lrp_individual
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


def compute_lrp_values(model, lrp_dataloader):
    negative_array, positive_array, pos_n_array, neg_n_array, \
    norm_array, pos_entropy_array, neg_entropy_array, R_array = [], [], [], [], [], [], [], []

    for data in lrp_dataloader:
        img, label = data
        R = lrp_individual(model, img.to(device), dataset, device=device)
        R_array.append(R)
        R_positive = np.maximum(R, 0)
        R_negative = np.minimum(R, 0)

        negative_array.append(sum(R_negative.flatten() < 0) / len(R_negative.flatten()))
        positive_array.append(sum(R_positive.flatten() > 0) / len(R_positive.flatten()))
        pos_n_array.append(np.linalg.norm(R_positive.flatten()))
        neg_n_array.append(np.linalg.norm(R_negative.flatten()))
        norm_array.append(np.linalg.norm(R.flatten()))
        pos_entropy_array.append(stats.entropy(R_positive.flatten()))
        neg_entropy_array.append(stats.entropy(R_negative.flatten()))

    add_item_to_csv('negative_lrp', dataset, negative_array)
    add_item_to_csv('positive_lrp', dataset, positive_array)
    add_item_to_csv('positive_lrp_norm', dataset, pos_n_array)
    add_item_to_csv('negative_lrp_norm', dataset, neg_n_array)
    add_item_to_csv('lrp_norm', dataset, norm_array)
    add_item_to_csv('positive_lrp_entropy', dataset, pos_entropy_array)
    add_item_to_csv('negative_lrp_entropy', dataset, neg_entropy_array)

    return torch.FloatTensor(np.array(R_array))


def add_item_to_csv(file_name, dataset, item, multiple=False):
    newpath = './csvs/{}/{}_{}.csv'.format(dataset, file_name, model_n)
    mode = 'a'
    if not os.path.exists(newpath):
        mode = 'w'
    with open(newpath, mode, newline='') as file:
        writer = csv.writer(file)
        if not multiple:
            writer.writerow([item])
        else:
            writer.writerows(item)


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
        model = ModelCIFAR10()
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
        model = Model()

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

    total_step = len(train_dataloader)

    lrp_step_size = 1

    lrp_image_dataset = CustomDataset(images, transform)
    lrp_image_dataset.transform = transform
    lrp_dataloader = DataLoader(lrp_image_dataset, batch_size=1)
    lrp_test_dataloader = DataLoader(lrp_image_dataset, batch_size=len(lrp_image_dataset))

    newpath = './results/{}'.format(dataset)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = './csvs/{}'.format(dataset)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = './lrp/{}'.format(dataset)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    lrp_per_class = {classname: [] for classname in images['label_int']}
    accuracy_per_class = {classname: [] for classname in images['label_int']}
    R = []
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
                add_item_to_csv('train_loss_class_{}'.format(train_set.class_to_idx[classname]), dataset, train_loss_class)

            if (i+1) % lrp_step_size == 0:
                # print statistics
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Running Loss: {:.4f}'.format(
                    epoch, epoch, i + 1, total_step, loss.item(), running_loss/(i+1)))

                add_item_to_csv('running_loss', dataset, running_loss/(i+1))

                # lrp values
                model.eval()
                R_values = compute_lrp_values(model, lrp_dataloader)
                R.append(R_values)

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

                correct_lrp = 0
                total_lrp = 0
                testing_loss_lrp = 0.0

                correct_pred_lrp = {classname: 0 for classname in test_set.classes}
                total_pred_lrp = {classname: 0 for classname in test_set.classes}

                with torch.no_grad():
                    for data_lrp in lrp_test_dataloader:
                        inputs_lrp, labels_lrp = data_lrp[0].to(device), data_lrp[1].to(device)
                        # calculate outputs by running images through the network
                        outputs_lrp = model(inputs_lrp.float())
                        testing_loss_lrp += criterion(outputs_lrp, labels_lrp).item()
                        add_item_to_csv('test_lrp_loss', dataset, criterion(outputs_lrp, labels_lrp).item())
                        loss_all = criterion_all(outputs_lrp, labels_lrp).detach().cpu().numpy()
                        add_item_to_csv('test_lrp_loss_all', dataset, loss_all)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted_lrp = torch.max(outputs_lrp.data, 1)
                        total_lrp += labels_lrp.size(0)
                        correct_lrp += (predicted_lrp == labels_lrp).sum().item()
                        for label, prediction in zip(labels_lrp, predicted_lrp):
                            if label == prediction:
                                correct_pred_lrp[test_set.classes[label]] += 1
                            total_pred_lrp[test_set.classes[label]] += 1

                add_item_to_csv('test_accuracy', dataset, 100*correct / total)

                add_item_to_csv('test_lrp_accuracy', dataset, 100*correct_lrp / total_lrp)

                for classname, correct_count in correct_pred.items():
                    accuracy = float(correct_count) / total_pred[classname]
                    accuracy_per_class[test_set.class_to_idx[classname]].append(accuracy)
                    add_item_to_csv('test_accuracy_class_{}'.format(test_set.class_to_idx[classname]), dataset, accuracy)

                for classname, correct_count in correct_pred_lrp.items():
                    accuracy = float(correct_count) / total_pred_lrp[classname]
                    lrp_per_class[test_set.class_to_idx[classname]].append(accuracy)
                    add_item_to_csv('test_accuracy_lrp_class_{}'.format(test_set.class_to_idx[classname]), dataset, accuracy)

                for classname, loss in test_loss_per_class.items():
                    loss = float(loss) / total_pred[classname]
                    add_item_to_csv('test_loss_class_{}'.format(test_set.class_to_idx[classname]), dataset, loss)

    print('Finished Training')

    if model_n == 3:
        torch.save(torch.FloatTensor(torch.stack(R)),
               'lrp/{}/lrp_values_model_{}'.format(dataset, model_n))
        torch.save(model.state_dict(), "./lrp/model_{}_{}.ckpt".format(model_n, dataset))


if __name__ == "__main__":
    train()
