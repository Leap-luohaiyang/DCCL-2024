import os
import argparse
import collections
import random
import warnings
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
from Datasets.data_pre import load_data_indiana, get_source_data, get_target_data
from contrastive_loss import SupConLoss, CDCSourceAnchor, CDCTargetAnchor
from utils import *
from Model.classifier import Classifier
from Model.feature_extractor import FDSSCNetwork

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='Hyperspectral Image Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


dset_classes = ['Concrete/Asphalt',
                'Corn-CleanTill',
                'Corn-CleanTill-EW',
                'Orchard',
                'Soybeans-CleanTill',
                'Soybeans-CleanTill-EW',
                'Wheat']
classes_acc = {}
for i in dset_classes:
    classes_acc[i] = []
    classes_acc[i].append(0)
    classes_acc[i].append(0)

num_epoch = args.epochs
BATCH_SIZE = args.batch_size
HalfWidth = 2
nBand = 220
patch_size = 2 * HalfWidth + 1
CLASS_NUM = 7
K_NUM = 3
threshold = 0.2

file_path = '/tmp/DA_HSIC/PublicDataset/Indiana/DataCube.mat'

set_seed(876)

source_data, target_data, source_label, target_label = load_data_indiana(file_path)

train_data_s, train_label_s = get_source_data(source_data, source_label, HalfWidth, 180)  # (1260, 9, 9, 102)
test_data, test_label, test_gt, RandPerm, Row, Column = get_target_data(target_data, target_label, HalfWidth)

train_dataset_s = TensorDataset(torch.tensor(train_data_s).unsqueeze(1), torch.tensor(train_label_s))
train_dataset_t = TensorDataset(torch.tensor(test_data).unsqueeze(1), torch.tensor(test_label))
test_dataset = TensorDataset(torch.tensor(test_data).unsqueeze(1), torch.tensor(test_label))

train_loader_s = DataLoader(train_dataset_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
train_loader_t = DataLoader(train_dataset_t, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

loader_s_no_drop = DataLoader(train_dataset_s, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
loader_t_no_drop = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

len_source_loader = len(train_loader_s)
len_target_loader = len(train_loader_t)

G = FDSSCNetwork(band=nBand).cuda()
F1 = Classifier(G.output_num(), 1000, CLASS_NUM).cuda()

optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
optimizer_f = optim.SGD(list(F1.parameters()), momentum=0.9, lr=args.lr, weight_decay=0.0005)


def train(total_epochs):
    criterion = nn.CrossEntropyLoss().cuda()
    cont_sa = CDCSourceAnchor(temperature=0.5).cuda()
    cont_ta = CDCTargetAnchor(temperature=0.5).cuda()
    cont_ss = SupConLoss(temperature=0.5).cuda()
    best_acc = 0

    for ep in range(total_epochs):
        """Mapped source class centers"""
        mapped_source_class_centers, features_all, labels_all = source_class_centers(loader_s_no_drop, G, CLASS_NUM, G.output_num())
        mapped_source_class_centers = mapped_source_class_centers.cuda()
        '''
        mapped_source_class_centers: The feature vector of each class center in source domain
        e.g mapped_source_class_centers[0]: The feature vector of the tree class center in source domain
        '''

        """Mapped target cluster centers"""
        mapped_target_cluster_centers = target_cluster_centers(loader_t_no_drop, G, CLASS_NUM,
                                                               G.output_num(), mapped_source_class_centers)
        mapped_target_cluster_centers = mapped_target_cluster_centers.cuda()
        '''
        mapped_target_cluster_centers: The cluster center of each class obtained by K-means clustering of all the feature vectors in target domain
        '''

        """Mapped target class centers"""
        mapped_target_class_centers = target_class_centers(mapped_source_class_centers, mapped_target_cluster_centers)
        mapped_target_class_centers = mapped_target_class_centers.cuda()
        '''
        mapped_target_class_centers: The class centers in target domain are obtained by one-to-one matching with the class centers in source domain
        e.g mapped_target_class_centers[0]: The feature vector of the tree class center in target domain
        '''

        for batch_idx, data in enumerate(zip(train_loader_s, train_loader_t)):
            G.train()
            F1.train()

            (img_s, label_s), (img_t, label_t) = data

            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = label_s.cuda()

            data_all = torch.cat((img_s, img_t), 0)
            bs = label_s.shape[0]

            """select high-confidence target samples"""
            feature_all = G(data_all)
            features_s = feature_all[:bs, :]
            features_t = feature_all[bs:, :]
            predict_all = F1(feature_all)
            predict_s = predict_all[:bs, :]

            nearest_target_center_labels = nearest_class_center(features_t, mapped_target_class_centers)
            sample_labels = nearest_source_examples(features_t, features_all, labels_all, G, K_NUM)
            label_vector = torch.cat((sample_labels, nearest_target_center_labels.view(BATCH_SIZE, 1)), dim=1)
            '''
            nearest_target_center_labels: The label assigned according to the nearest class center in target domain 
            sample_labels: The label assigned according to the K nearest examples in source domain 
            label_vector: 1+K pseudo labels
            '''

            unique_counts = torch.tensor([len(torch.unique(row)) for row in label_vector])  # 计算每一行元素的唯一值数量
            consistent_indices = torch.nonzero(unique_counts == 1, as_tuple=False).squeeze()  # 唯一值数量为 1 的行索引
            if consistent_indices.shape == torch.Size([0]):
                consistent_labels = torch.tensor([]).cuda()
            else:
                if consistent_indices.dim() == 0:
                    consistent_indices = torch.unsqueeze(consistent_indices, 0)
                consistent_labels = label_vector[consistent_indices, 0]
            '''
            consistent_indices: Indices of the high confidence examples in target domain 
            '''

            bvsb_indices, target_center_bvsb_labels = nearest_target_center_bvsb(features_t,
                                                                                 mapped_target_class_centers, threshold)

            total_indices, total_pseudo_label = integrated_con_bvsb(consistent_indices, consistent_labels, bvsb_indices,
                                                                    target_center_bvsb_labels)

            """perform contrastive learning"""
            if total_indices.shape == torch.Size([0]):
                cdc_loss = 0
                idc_loss = cont_ss(features_s, label_s)
            else:
                selected_feature_t = features_t[total_indices]
                cdc_loss = cont_sa(features_s, selected_feature_t, label_s, total_pseudo_label)
                cdc_loss += cont_ta(features_s, selected_feature_t, label_s, total_pseudo_label)
                idc_loss = cont_ss(features_s, label_s)

            """back-propagate and update network"""
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            cls_loss = criterion(predict_s, label_s)
            all_loss = cls_loss + 0.2 * cdc_loss + 0.2 * idc_loss
            all_loss.backward()

            optimizer_g.step()
            optimizer_f.step()

        # test
        current_acc = test(ep + 1)
        if current_acc > best_acc:
            best_acc = current_acc
            print('save model with acc ' + str(best_acc))

        print('util now the best acc is ' + str(best_acc))


def test(epoch):
    G.eval()
    F1.eval()
    correct_add = 0
    pred_test = []
    gt_test = []
    size = 0
    print('-' * 100, '\nTesting')
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img, label = data
            img, label = img.cuda(), label.long().cuda()
            feature = G(img)
            output = F1(feature)
            pred = output.data.max(1)[1]
            pred_test.extend(np.array(pred.cpu()))
            gt_test.extend(np.array(label.cpu()))
            correct_add += pred.eq(label.data).cpu().sum()
            size += label.data.size()[0]
            for i in range(label.shape[0]):
                key_label = dset_classes[label.long()[i].item()]
                key_pred = dset_classes[pred.long()[i].item()]
                classes_acc[key_label][1] += 1
                if key_pred == key_label:
                    classes_acc[key_pred][0] += 1

    print('Epoch: {:d}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, correct_add, size, 100. * float(correct_add) / size))
    avg = []
    for i in dset_classes:
        print('\t{}: [{}/{}] ({:.6f}%)'.format(i, classes_acc[i][0], classes_acc[i][1],
                                               100. * classes_acc[i][0] / classes_acc[i][1]))
        avg.append(float(classes_acc[i][0]) / classes_acc[i][1])
    for i in dset_classes:
        classes_acc[i][0] = 0
        classes_acc[i][1] = 0

    collections.Counter(pred_test)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)
    print('\tOA: ', float(correct_add) / size)
    print('\tAA: ', np.average(avg))
    print('\tKappa: ', kappa)

    return float(correct_add) / size


train(args.epochs + 1)
