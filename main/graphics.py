import math
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn import preprocessing
from scipy import stats
import csv
from visualization_values import *


method='lrp'
dataset='FashionMNIST'
path = 'master_thesis_csvs/{}/csvs/{}/'.format(method, dataset)
path_plot = 'results/{}/'.format(dataset)


def scatter_plot(x,y,xlabel,ylabel,title):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def normal_plot(plot_array, label_array, title, ylabel, savefig=''):
    for i, plot in enumerate(plot_array):
        plt.plot(plot, label=label_array[i])
    plt.title(title)
    plt.xlabel('time step')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path_plot + savefig)
    plt.show()


def normal_plot_2(x, y, xlabel, ylabel, title):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(path_plot + title)
    plt.show()


def simple_read_csv(path_csv):
    trend_array = []
    array_scatter = []
    with open(path + path_csv, 'r') as file:
        csvreader = csv.reader(file)
        for i, row in enumerate(csvreader):
            if i % 1 == 0:
                array_scatter.append(float(row[-1]))
                trend_array.append(np.mean(array_scatter))
    return trend_array, array_scatter


def read_csv_attr(path_csv):
    trend_array = []
    array_scatter = []
    with open(path + path_csv, 'r') as file:
        csvreader = csv.reader(file)
        prop = 0
        for i, row in enumerate(csvreader):
            if i % 1 == 0:
                attr_values_str = row[0][1:-1]
                attr_values_array = np.fromstring(attr_values_str, dtype=float, count=-1, sep=',')
                attr_values_array = [0 if math.isnan(x) else x for x in attr_values_array]
                prop += np.mean(attr_values_array)
                array_scatter.append(np.mean(attr_values_array))
                trend_array.append(prop/(len(trend_array)+1))
    return trend_array, array_scatter


def read_csv_per_class(path_csv, c, sep=','):
    trend_array = []
    array_scatter = []
    with open(path + path_csv, 'r') as file:
        csvreader = csv.reader(file)
        prop = 0
        for i, row in enumerate(csvreader):
            if i % 1 == 0:
                attr_values_str = row[0][1:-1]
                attr_values_array = np.fromstring(attr_values_str, dtype=float, count=-1, sep=sep)
                attr_values_array = [0 if math.isnan(x) else x for x in attr_values_array][0+20*c:20+20*c]
                prop += np.mean(attr_values_array)
                array_scatter.append(np.mean(attr_values_array))
                trend_array.append(prop/(len(trend_array)+1))
    return trend_array, array_scatter


for k in range(3,6):

    train_loss, train_loss_scatter = simple_read_csv("train_loss_{}.csv".format(k))
    test_loss, test_loss_scatter = simple_read_csv("test_loss_{}.csv".format(k))
    attrib_loss, attrib_loss_scatter = simple_read_csv("test_{}_loss_{}.csv".format(method, k))

    test_accuracy, test_accuracy_scatter = simple_read_csv("test_accuracy_{}.csv".format(k))
    train_accuracy, train_accuracy_scatter = simple_read_csv("train_accuracy_{}.csv".format(k))
    test_attr_accuracy, test_attr_accuracy_scatter = simple_read_csv("test_{}_accuracy_{}.csv".format(method, k))

    positive_gradcam, positive_gradcam_scatter = read_csv_attr("positive_{}_{}.csv".format(method, k))
    negative_gradcam, negative_gradcam_scatter = read_csv_attr("negative_{}_{}.csv".format(method, k))

    positive_gradcam_norm, positive_gradcam_norm_scatter = read_csv_attr("positive_{}_norm_{}.csv".format(method, k))
    negative_gradcam_norm, negative_gradcam_norm_scatter = read_csv_attr("negative_{}_norm_{}.csv".format(method, k))
    negative_gradcam_entropy, negative_gradcam_entropy_scatter = read_csv_attr("negative_{}_entropy_{}.csv".format(method, k))
    positive_gradcam_entropy, positive_gradcam_entropy_scatter = read_csv_attr("positive_{}_entropy_{}.csv".format(method, k))

    for i in range(10):
        attr_acc_trend, attr_acc_scatter = simple_read_csv('test_accuracy_{}_class_{}_{}.csv'.format(method, i, k))
        test_acc_trend, test_acc_scatter = simple_read_csv('test_accuracy_class_{}_{}.csv'.format(i, k))
        train_acc_trend, train_acc_scatter = simple_read_csv('train_accuracy_class_{}_{}.csv'.format(i, k))
        normal_plot([train_acc_trend, test_acc_trend, attr_acc_trend], ['train accuracy', 'test accuracy', 'test attribution accuracy'],
                    'Trend of the accuracy for class {} for {}'.format(i, dataset), 'accuracy', 'accuracy_class_{}_{}'.format(i, dataset))

        attr_loss_trend, attr_loss_scatter = read_csv_per_class('test_{}_loss_all_{}.csv'.format(method, k), i, sep=' ')
        test_loss_trend, test_loss_scatter = simple_read_csv('test_loss_class_{}_{}.csv'.format(i, k))
        train_loss_trend, train_loss_scatter = simple_read_csv('train_loss_class_{}_{}.csv'.format(i, k))
        normal_plot([train_loss_trend, test_loss_trend, attr_loss_trend],
                    ['train loss', 'test loss', 'test attribution loss'],
                    'Trend of the loss for class {} for {}'.format(i, dataset), 'loss', 'loss_class_{}_{}'.format(i, dataset))

        pos_entr, pos_entr_scatter = read_csv_per_class("positive_{}_entropy_{}.csv".format(method, k), i)
        neg_entr, neg_entr_scatter = read_csv_per_class("negative_{}_entropy_{}.csv".format(method, k), i)
        pos_norm, pos_norm_scatter = read_csv_per_class("positive_{}_norm_{}.csv".format(method, k), i)
        neg_norm, neg_norm_scatter = read_csv_per_class("negative_{}_norm_{}.csv".format(method, k), i)

        pos_prop, pos_prop_scatter = read_csv_per_class("positive_{}_{}.csv".format(method, k), i)
        neg_prop, neg_prop_scatter = read_csv_per_class("negative_{}_{}.csv".format(method, k), i)

        normal_plot([pos_prop, neg_prop], ['positive proportion', 'negative proportion'],
                    'Proportion of the attribution scores for class {} for {}'.format(i, dataset), 'proportion', 'proportion_class_{}_{}'.format(i, dataset))
        normal_plot([pos_norm, neg_norm],
                    ['positive norm', 'negative norm'], 'Norm of the attribution scores for class {} for {}'.format(i, dataset), 'norm', 'norm_class_{}_{}'.format(i, dataset))
        normal_plot([pos_entr, neg_entr], ['positive entropy', 'negative entropy'],
                    'Entropy of the attribution scores for class {} for {}'.format(i, dataset), 'entropy', 'entropy_class_{}_{}'.format(i, dataset))
        scaler = preprocessing.MinMaxScaler()
        train_loss_scatter = scaler.fit_transform(
            np.array(train_loss_scatter).reshape(len(train_loss_scatter), 1)).reshape(-1)
        test_loss_scatter = scaler.fit_transform(
            np.array(test_loss_scatter).reshape(len(test_loss_scatter), 1)).reshape(-1)
        attrib_loss_scatter = scaler.fit_transform(
            np.array(attr_loss_scatter).reshape(len(attr_loss_scatter), 1)).reshape(-1)
        positive_gradcam_entropy_scatter = scaler.fit_transform(
            np.array(pos_entr_scatter).reshape(len(pos_entr_scatter), 1)).reshape(-1)
        negative_gradcam_entropy_scatter = scaler.fit_transform(
            np.array(neg_entr_scatter).reshape(len(neg_entr_scatter), 1)).reshape(-1)
        positive_gradcam_norm_scatter = scaler.fit_transform(
            np.array(pos_norm_scatter).reshape(len(pos_norm_scatter), 1)).reshape(-1)
        negative_gradcam_norm_scatter = scaler.fit_transform(
            np.array(neg_norm_scatter).reshape(len(neg_norm_scatter), 1)).reshape(-1)
        test_accuracy_scatter = scaler.fit_transform(
            np.array(test_acc_scatter).reshape(len(test_acc_scatter), 1)).reshape(-1)
        train_accuracy_scatter = scaler.fit_transform(
            np.array(train_acc_scatter).reshape(len(train_acc_scatter), 1)).reshape(-1)
        test_attr_accuracy_scatter = scaler.fit_transform(
            np.array(attr_acc_scatter).reshape(len(attr_acc_scatter), 1)).reshape(-1)

        summmary_measures_scatter = [positive_gradcam_norm_scatter, negative_gradcam_norm_scatter,
                                     positive_gradcam_entropy_scatter,
                                     negative_gradcam_entropy_scatter]
        perfor_measures_scatter = [test_loss_scatter, train_loss_scatter, attrib_loss_scatter, test_accuracy_scatter,
                                   train_accuracy_scatter, test_attr_accuracy_scatter]

        cell_text = []
        columns = ['positive norm', 'negative norm', 'positive entropy', 'negative entropy']
        rows = ['test loss', 'train loss', 'attribution loss', 'test accuracy', 'train accuracy',
                'test attribution accuracy']
        fig = plt.figure(figsize=(40.00, 40))
        ax = fig.subplots(6, 4)
        for k in range(len(perfor_measures_scatter)):
            text = []
            for j in range(len(summmary_measures_scatter)):
                rho, p = stats.spearmanr(perfor_measures_scatter[k], summmary_measures_scatter[j])
                text.append('rho: %.3f, p: %.2f' % (rho, p))
                ax[k, j].scatter(range(len(perfor_measures_scatter[k])), perfor_measures_scatter[k])
                ax[k, j].scatter(range(len(summmary_measures_scatter[j])), summmary_measures_scatter[j])
                ax[0, j].set_title(columns[j])
                ax[k, 0].set(ylabel=rows[k])
                ax[k, j].legend([rows[k], columns[j]])
            cell_text.append(text)
        fig.suptitle(dataset, fontsize=16)
        plt.savefig(path_plot + '{}_{}'.format(i, dataset))
        plt.show()

        fig = plt.figure(figsize=(20.00, 10))
        axs = fig.subplots(1, 1)
        axs.axis('tight')
        axs.axis('off')
        the_table = axs.table(cellText=cell_text,
                              rowLabels=rows,
                              colLabels=columns,
                              loc='center')
        plt.savefig(path_plot + 'Spearman rho {} {}'.format(i, dataset))
        plt.show()
   
    normal_plot([train_loss, test_loss, attrib_loss], ['train loss', 'test loss', 'test attribution loss'], 'Trend of the loss for {}'.format(dataset), 'loss')
    normal_plot([train_accuracy, test_accuracy, test_attr_accuracy], ['train accuracy', 'test accuracy', 'test attribution accuracy'], 'Trend of the accuracy for {}'.format(dataset), 'accuracy')

    normal_plot([positive_gradcam, negative_gradcam], ['positive proportion', 'negative proportion'],
                'Proportion of the attribution scores for {}'.format(dataset), 'proportion')
    normal_plot([positive_gradcam_norm, negative_gradcam_norm],
                ['positive norm', 'negative norm'], 'Norm of the attribution scores for {}'.format(dataset), 'norm')
    normal_plot([positive_gradcam_entropy, negative_gradcam_entropy], ['positive entropy', 'negative entropy'],
                'Entropy of the attribution scores for {}'.format(dataset), 'entropy')
    
    scaler = preprocessing.MinMaxScaler()
    train_loss_scatter = scaler.fit_transform(np.array(train_loss_scatter).reshape(len(train_loss_scatter), 1)).reshape(-1)
    test_loss_scatter = scaler.fit_transform(np.array(test_loss_scatter).reshape(len(test_loss_scatter), 1)).reshape(-1)
    attrib_loss_scatter = scaler.fit_transform(np.array(attrib_loss_scatter).reshape(len(attrib_loss_scatter), 1)).reshape(-1)
    positive_gradcam_entropy_scatter = scaler.fit_transform(
        np.array(positive_gradcam_entropy_scatter).reshape(len(positive_gradcam_entropy_scatter), 1)).reshape(-1)
    negative_gradcam_entropy_scatter = scaler.fit_transform(
        np.array(negative_gradcam_entropy_scatter).reshape(len(negative_gradcam_entropy_scatter), 1)).reshape(-1)
    positive_gradcam_norm_scatter = scaler.fit_transform(
        np.array(positive_gradcam_norm_scatter).reshape(len(positive_gradcam_norm_scatter), 1)).reshape(-1)
    negative_gradcam_norm_scatter = scaler.fit_transform(
        np.array(negative_gradcam_norm_scatter).reshape(len(negative_gradcam_norm_scatter), 1)).reshape(-1)
    test_accuracy_scatter = scaler.fit_transform(np.array(test_accuracy_scatter).reshape(len(test_accuracy_scatter), 1)).reshape(-1)
    train_accuracy_scatter = scaler.fit_transform(np.array(train_accuracy_scatter).reshape(len(train_accuracy_scatter), 1)).reshape(-1)
    test_attr_accuracy_scatter = scaler.fit_transform(np.array(test_attr_accuracy_scatter).reshape(len(test_attr_accuracy_scatter), 1)).reshape(-1)

    summmary_measures_scatter = [positive_gradcam_norm_scatter, negative_gradcam_norm_scatter,
                                 positive_gradcam_entropy_scatter,
                                 negative_gradcam_entropy_scatter]
    perfor_measures_scatter = [test_loss_scatter, train_loss_scatter, attrib_loss_scatter, test_accuracy_scatter,
                               train_accuracy_scatter, test_attr_accuracy_scatter]

    cell_text = []
    columns = ['positive norm', 'negative norm', 'positive entropy', 'negative entropy']
    rows = ['test loss', 'train loss', 'attribution loss', 'test accuracy', 'train accuracy',
            'test attribution accuracy']

    fig = plt.figure(figsize=(40.00, 40))
    ax = fig.subplots(6, 4)
    for k in range(len(perfor_measures_scatter)):
        text = []
        for j in range(len(summmary_measures_scatter)):
            rho, p = stats.spearmanr(perfor_measures_scatter[k], summmary_measures_scatter[j])
            text.append('rho: %.3f, p: %.2f' % (rho, p))
            ax[k, j].scatter(range(len(perfor_measures_scatter[k])), perfor_measures_scatter[k])
            ax[k, j].scatter(range(len(summmary_measures_scatter[j])), summmary_measures_scatter[j])
            ax[0, j].set_title(columns[j])
            ax[k, 0].set(ylabel=rows[k])
            ax[k, j].legend([rows[k], columns[j]])
        cell_text.append(text)
    fig.suptitle(dataset, fontsize=16)
    plt.savefig(path_plot + '{}_{}'.format(i, dataset))
    plt.show()

    fig = plt.figure(figsize=(20.00, 10))
    axs = fig.subplots(1, 1)
    axs.axis('tight')
    axs.axis('off')
    the_table = axs.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')
    plt.savefig(path_plot + 'Spearman rho {} {}'.format(i, dataset))
    plt.show()

    # Heatmaps
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
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,), )
                                        ])
        # prepare transforms standard to MNIST
        train_set = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                      transform=transform,
                                                      download=True)
        test_set = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                     transform=transform,
                                                     download=True)

    images = []
    for label, value in train_set.class_to_idx.items():
        images.extend([img for index, img in enumerate(train_set.data)
                                   if train_set.targets[index] == value][:10])
        images.extend([img for index, img in enumerate(test_set.data)
                                   if test_set.targets[index] == value][:10])
    if dataset == 'CIFAR10':
        images = [torch.from_numpy(img) for img in images]

    path_lrp_values = 'master_thesis_csvs/lrp/lrp/{}/'.format(dataset)
    lrp_values = torch.load(path_lrp_values + 'lrp_values_model_3')

    from matplotlib.colors import ListedColormap
    for j in range(0,200,20):
        img_per_class = []
        img_real = []
        fig = plt.figure(figsize=(30.00, 15))
        ax = fig.subplots(2, 5)
        for i in range(0, 6000, 600):
            lrps = lrp_values[i][j]
            heatmap = lrps

            # normalize the heatmap
            if torch.max(heatmap) != 0:
                heatmap /= torch.max(heatmap)

            img = images[j].numpy()
            heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
            img_real.append(img)
            img = np.array(heatmap)
            img_per_class.append(heatmap)
        for m in range(10):
            k = 0
            b = 10 * ((np.abs(img_per_class[m]) ** 3.0).mean() ** (1.0 / 3))
            my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
            my_cmap[:, 0:3] *= 0.85
            my_cmap = ListedColormap(my_cmap)
            if m > 4:
                k+=1
            ax[k][m % 5].imshow(img_real[m], cmap='gray')
            ax[k][m%5].imshow(img_per_class[m], cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
            ax[k][m%5].axis('off')
        plt.savefig('complex_model_class_{}_epoch_lrp.png'.format(int(j/20)))
        plt.show()