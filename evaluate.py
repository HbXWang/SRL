from modules import get_resnet, SRL_Network, contrastive_loss
from utils import to_numpy, cluster_acc, best_map
from torch.utils import data
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from utils import get_config, save_model, load_checkpoint
from datasets import load_datasets, SRL_Transforms
import argparse
import torch
import os


def evaluation(loader_eval, model_eval, epoch, is_draw=False):
    model_eval.eval()
    with torch.no_grad():
        hidden_feature = []
        label_train = []
        label_pre = []
        for batch_idx, (x, y) in enumerate(tqdm(loader_eval)):
            if torch.cuda.device_count() > 1:
                z, c = model_eval.module.forward_cluster(x.to(device))
            else:
                z, c = model_eval.forward_cluster(x.to(device))
            hidden_feature.append(z)
            label_train.append(y)
            label_pre.append(torch.argmax(c, dim=1))
        hidden_feature = to_numpy(torch.cat(hidden_feature, dim=0))
        label_train = to_numpy(torch.cat(label_train, dim=0))
        label_pre = to_numpy(torch.cat(label_pre, dim=0))

        kmeans = KMeans(init='k-means++', n_clusters=class_num, n_init=10).fit(hidden_feature)
        cls_index = kmeans.labels_
        acc = cluster_acc(cls_index, label_train.astype(np.int32))
        NMI = metrics.normalized_mutual_info_score(label_train, cls_index, average_method='arithmetic')
        ari = metrics.adjusted_rand_score(label_train, cls_index)
        print('\n| Kmeans ACC = {:6f} NMI = {:6f} ARI = {:6f}'.format(acc, NMI, ari))

        acc = cluster_acc(label_pre, label_train.astype(np.int32))
        NMI = metrics.normalized_mutual_info_score(label_train, label_pre, average_method='arithmetic')
        ari = metrics.adjusted_rand_score(label_train, label_pre)
        print('\n| Pre ACC = {:6f} NMI = {:6f} ARI = {:6f}'.format(acc, NMI, ari))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = get_config("./config/config.yaml", parser)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model
    base_model = get_resnet(args.base_model)
    model_save_dir = args.model_save_dir

    # dataset
    data_transform = SRL_Transforms(args.image_size, 0.5, args.mean, args.std)
    print('==> load dataset..')
    test_dataset, class_num = load_datasets(args.dataset, args.dataset_dir, data_transform.test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=args.workers)
    print("| Completed.")

    print('==> Building model..')
    model = SRL_Network(base_model, feature_dim=args.feature_dim, class_num=class_num, k_crop=args.k_crops)
    cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    reload_epoch = 2000
    print('==> Reload Model..')
    checkpoint = load_checkpoint(args, reload_epoch, device)
    reload_epoch = checkpoint['epoch'] + 1

    model_para = checkpoint['net']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in model_para.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    evaluation(test_loader, model, reload_epoch, is_draw=False)
