from utils import get_config, save_model, load_checkpoint, setup_seed
from modules import get_resnet, SRL_Network, contrastive_loss
from datasets import load_datasets, SRL_Transforms
from utils import to_numpy, cluster_acc

from torch_clustering import PyTorchKMeans
import torch.backends.cudnn as cudnn
from torch.utils import data
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import argparse
import torch
import sys
import os


def train_epoch(model_train, loader_train, optimizer_train, lambda_parameter, prior_p, ratio_beta):
    model_train.train()
    total_loss, loss_i, loss_c = 0, 0, 0
    label_train = []
    label_pre = []
    for _, (images, y) in enumerate(tqdm(loader_train)):
        images = [aug_img.to(device, non_blocking=True) for aug_img in images]
        z_i, z_means, z_sigma, c_i, c_j = model_train(images)

        loss_instance = criterion_instance(z_i, z_means, z_sigma, c_i, c_j, prior_p, ratio_beta)
        loss_cluster = criterion_cluster(c_i, c_j) * lambda_parameter

        loss = loss_instance + loss_cluster

        optimizer_train.zero_grad()
        loss.backward()
        optimizer_train.step()

        total_loss += loss.item()
        loss_i += loss_instance.item()
        loss_c += loss_cluster.item()

        with torch.no_grad():
            prior_p = momentum * prior_p + (1 - momentum) * torch.mean(torch.cat((c_i, c_j), dim=0), dim=0)

        label_train.append(y)
        label_pre.append(torch.argmax(c_i, dim=1))

    total_loss = total_loss / len(loader_train)
    loss_i = loss_i / len(loader_train)
    loss_c = loss_c / len(loader_train)

    label_train = to_numpy(torch.cat(label_train, dim=0))
    label_pre = to_numpy(torch.cat(label_pre, dim=0))

    ari = metrics.adjusted_rand_score(label_train, label_pre)
    acc = cluster_acc(label_pre, label_train.astype(np.int32))
    nmi = metrics.normalized_mutual_info_score(label_train, label_pre, average_method='arithmetic')

    return total_loss, loss_i, loss_c, acc, nmi, ari, prior_p


def evaluation(loader_test, model_test):
    model_test.eval()
    with torch.no_grad():
        hidden_feature = []
        label_train = []
        label_pre = []
        for batch_idx, (x, y) in enumerate(tqdm(loader_test)):
            if torch.cuda.device_count() > 1:
                z, c = model_test.module.forward_cluster(x.to(device))
            else:
                z, c = model_test.forward_cluster(x.to(device))
            hidden_feature.append(z)
            label_train.append(y)
            label_pre.append(torch.argmax(c, dim=1))
        # hidden_feature = to_numpy(torch.cat(hidden_feature, dim=0))
        label_train = to_numpy(torch.cat(label_train, dim=0))
        label_pre = to_numpy(torch.cat(label_pre, dim=0))

        # cosine
        hidden_feature = torch.cat(hidden_feature, dim=0)
        kwargs = {'metric': 'euclidean', 'distributed': False, 'random_state': 0, 'n_clusters': class_num,
                  'verbose': False}
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        cls_index = clustering_model.fit_predict(hidden_feature)

        cls_index = to_numpy(cls_index)
        acc = cluster_acc(cls_index, label_train.astype(np.int32))
        nmi = metrics.normalized_mutual_info_score(label_train, cls_index, average_method='arithmetic')
        ari = metrics.adjusted_rand_score(label_train, cls_index)
        print('\n| Kmeans ACC = {:6f} NMI = {:6f} Bal = {:6f}'.format(acc, nmi, ari))

        acc = cluster_acc(label_pre, label_train.astype(np.int32))
        nmi = metrics.normalized_mutual_info_score(label_train, label_pre, average_method='arithmetic')
        ari = metrics.adjusted_rand_score(label_train, label_pre)
        print('\n| Pre ACC = {:6f} NMI = {:6f} ARI = {:6f}'.format(acc, nmi, ari))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = get_config("config/config.yaml", parser)

    # setting
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    data_transform = SRL_Transforms(args.image_size, 0.5, args.mean, args.std, k_crops=args.k_crops)
    print('==> load dataset..')
    test_dataset, class_num = load_datasets(args.dataset, args.dataset_dir, data_transform.test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=args.workers)
    train_dataset, _ = load_datasets(args.dataset, args.dataset_dir, data_transform)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                   num_workers=args.workers)
    print("| Completed.")

    # set model
    print('==> Building model..')
    base_model = get_resnet(args.base_model)
    model = SRL_Network(base_model, feature_dim=args.feature_dim, class_num=class_num, k_crop=args.k_crops)
    cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    print('==> Begin Train..')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=5e-4, nesterov=False, dampening=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [600, 950, 1300, 1650], gamma=0.1)
    criterion_instance = contrastive_loss.Local_structure_clustering(args.batch_size,
                                                                     args.local_structure_driven_temperature, device).to(device)
    criterion_cluster = contrastive_loss.Global_structure_clustering(class_num, args.global_structure_driven_temperature).to(device)

    reload_epoch = args.start_epoch
    if args.reload:
        print('==> Reload Model..')
        checkpoint = load_checkpoint(args, reload_epoch, device)
        reload_epoch = checkpoint['epoch'] + 1

        model_para = checkpoint['net']
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in model_para.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    momentum = torch.tensor(0.99).to(device)
    prior = torch.ones(class_num).to(device) / class_num

    sys.stdout = open('./results/results.log', mode='w', encoding='utf-8')
    for epoch in tqdm(range(args.start_epoch, args.epoch_num + 1)):

        ratio = args.lambda_l * ((epoch + 1) * 1.0 / args.epoch_num)
        loss_epoch, loss_i_epoch, loss_c_epoch, acc_epoch, nmi_epoch, ari_epoch, prior = \
            train_epoch(model, train_loader, optimizer, args.lambda_parameter, prior, ratio)
        lr_scheduler.step()

        print(
            "\nEpoch [{}/{}]\t Total Loss: {:.4f}\t Instance Loss: {:.4f}\t Cluster Loss: {:.4f}\t ACC = {:4f}\t NMI = {:4f}\t ARI = {:4f}"
            .format(epoch, args.epoch_num, loss_epoch, loss_i_epoch, loss_c_epoch, acc_epoch, nmi_epoch, ari_epoch))

        if epoch % 50 == 0:
            save_model(args, model, optimizer, epoch)
            print('\n==> Testing：{}/{}..'.format(epoch, args.epoch_num))
            evaluation(test_loader, model)
