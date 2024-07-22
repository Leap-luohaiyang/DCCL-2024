import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np


def source_class_centers(loader, net, class_num, feature_dim):
    """
    :param loader: dataloader of source dataset
    :param net: feature extractor
    :param class_num: total number of class
    :param feature_dim: dimensions of feature vectors
    :return: mapped source class center, features and labels of all the source domain examples
    """
    net.eval()
    class_centers = torch.zeros((class_num, feature_dim))
    all_features = torch.empty(0, feature_dim)
    all_features = all_features.cuda()
    all_labels = torch.empty(0)
    all_labels = all_labels.cuda()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            feature = net(img)  # (batch_size, feature_dim)
            all_features = torch.cat((all_features, feature))
            all_labels = torch.cat((all_labels, label))

    for i in range(class_num):
        class_indices = torch.nonzero(torch.eq(all_labels, i), as_tuple=False).squeeze()
        class_centers[i] = torch.mean(all_features[class_indices], dim=0)

    class_centers = F.normalize(class_centers, p=2, dim=1)

    return class_centers, all_features, all_labels


def target_cluster_centers(loader, net, class_num, feature_dim, init_centers):
    """
    :param loader: dataloader of target dataset
    :param net: feature extractor
    :param class_num: total number of class
    :param feature_dim: dimensions of feature vectors
    :param init_centers: the cluster centers are initialised with class prototypes
    :return: mapped target cluster centers
    """
    net.eval()
    all_features = torch.empty(0, feature_dim)
    all_features = all_features.cuda()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            img, _ = data
            img = img.cuda()
            feature = net(img)
            all_features = torch.cat((all_features, feature), dim=0)

    all_features = F.normalize(all_features, p=2, dim=1)
    all_features = all_features.data.cpu().numpy()
    init_centers = init_centers.data.cpu().numpy()

    kmeans = KMeans(n_clusters=class_num, init=init_centers)
    kmeans.fit(all_features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers)
    cluster_centers = F.normalize(cluster_centers, p=2, dim=1)

    return cluster_centers


def target_class_centers(class_centers_s, cluster_centers_t):
    """
    :param class_centers_s: mapped source class centers
    :param cluster_centers_t: mapped target cluster centers tensor
    :return: mapped target class centers
    """
    class_centers_s = class_centers_s.cpu().numpy()
    cluster_centers_t = cluster_centers_t.cpu().numpy()
    dis_matrix = cdist(class_centers_s, cluster_centers_t, 'euclidean')
    '''
    dis_matrix: numpy.array (class_num, class_num)
    The distance matrix between each class center in source domain and cluster center in target domain obtained by K-means
    '''

    _, target_cluster_ind = linear_sum_assignment(dis_matrix)
    '''
    source_class_ind: [0 1 2 3 4 5 6]
    target_cluster_ind: [~] The number of the cluster centers in target domain that matches each class center in source domain
    '''
    class_centers_t = cluster_centers_t[target_cluster_ind]
    class_centers_t = torch.tensor(class_centers_t)

    return class_centers_t


def nearest_source_examples(feature_t, feature_s_all, label_s_all, net, k_num):
    """
    :param feature_t: feature vector of target data (batch_size, feature_dim)
    :param feature_s_all: features of all the source domain examples
    :param label_s_all: labels of all the source domain examples
    :param net: feature extractor
    :param k_num: k nearest source example
    :return: K nearest source examples for mining the example information
    """
    net.eval()
    nor_feature_t = F.normalize(feature_t, p=2, dim=1)
    nor_feature_s = F.normalize(feature_s_all, p=2, dim=1)

    dis_matrix = torch.cdist(nor_feature_t, nor_feature_s, p=2).squeeze(0)
    _, nearest_indices = torch.topk(dis_matrix, k=k_num, dim=1, largest=False)
    nearest_examples = label_s_all[nearest_indices]

    return nearest_examples


def nearest_target_center_bvsb(feature_t, class_centers_t, distance_threshold):
    """
    :param feature_t: feature vector of target data
    :param class_centers_t: mapped target class centers
    :param distance_threshold: distance threshold for excluding ambiguous pseudo-labels
    :return: nearest class centers for mining the center information
    """
    nor_feature = F.normalize(feature_t, p=2, dim=1)
    dis_matrix = torch.cdist(nor_feature, class_centers_t, p=2).squeeze(0)

    sorted_matrix, _ = torch.sort(dis_matrix, dim=1)
    min_values = sorted_matrix[:, 0]
    second_min_values = sorted_matrix[:, 1]
    diff = second_min_values - min_values

    selected_indices = torch.nonzero(diff > distance_threshold, as_tuple=False).squeeze()
    if selected_indices.shape == torch.Size([0]):
        min_indices = torch.tensor([]).cuda()
    else:
        if selected_indices.dim() == 0:
            selected_indices = torch.unsqueeze(selected_indices, 0)
        filter_matrix = dis_matrix[selected_indices]
        min_indices = torch.argmin(filter_matrix, dim=1)

    return selected_indices, min_indices


def nearest_class_center(feature_t, class_centers):
    """
    :param feature_t: feature vector of target data
    :param class_centers: mapped target class centers
    :return: nearest class centers for mining the center information
    """
    nor_feature = F.normalize(feature_t, p=2, dim=1)
    dis_matrix = torch.cdist(nor_feature, class_centers, p=2).squeeze(0)
    nearest_indices = torch.argmin(dis_matrix, dim=1)

    return nearest_indices


def integrated_con_bvsb(indices_con, pseudo_label_con, indices_bvsb, pseudo_label_bvsb):
    indices_con = indices_con.cpu().numpy()
    pseudo_label_con = pseudo_label_con.cpu().numpy()
    indices_bvsb = indices_bvsb.cpu().numpy()
    pseudo_label_bvsb = pseudo_label_bvsb.cpu().numpy()

    indices_inter = np.intersect1d(indices_con, indices_bvsb)
    unique_indices_con = np.setdiff1d(indices_con, indices_bvsb)
    unique_indices_bvsb = np.setdiff1d(indices_bvsb, indices_con)

    index_inter_indices_con = np.where(np.isin(indices_con, indices_inter))[0]
    # index_inter_indices_bvsb = np.where(np.isin(indices_bvsb, indices_inter))[0]
    index_unique_indices_con = np.where(np.isin(indices_con, unique_indices_con))[0]
    index_unique_indices_bvsb = np.where(np.isin(indices_bvsb, unique_indices_bvsb))[0]

    pseudo_label_inter = pseudo_label_con[index_inter_indices_con]
    pseudo_label_unique_con = pseudo_label_con[index_unique_indices_con]
    pseudo_label_unique_bvsb = pseudo_label_bvsb[index_unique_indices_bvsb]

    all_indices = np.concatenate((indices_inter, unique_indices_con, unique_indices_bvsb))
    all_pseudo_label = np.concatenate((pseudo_label_inter, pseudo_label_unique_con, pseudo_label_unique_bvsb))
    all_indices = torch.tensor(all_indices)
    all_pseudo_label = torch.tensor(all_pseudo_label).cuda()

    return all_indices, all_pseudo_label
