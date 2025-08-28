from __future__ import print_function
from numpy import *
import torch
import torch.optim as optim
from my_FSL_MIL_GNN.models.MIL_AMGNN_FSL_new10 import AMGraphBased28x28x1
from sklearn.cluster import AgglomerativeClustering
import pickle
from sklearn import metrics
import sys
import random
import time
import argparse
from torch.utils.data import DataLoader

bags = 64


def cast_cuda(input):

    if type(input) == type([]):
        for i in range(len(input)):
            input[i] = cast_cuda(input[i])
    else:
        return input.cuda()
    return input


def get_task_batch(batch_idx, train_loader):

    if batch_idx!=[]:
        index = batch_idx
        train_data_list = [i for i in range(len(train_loader.dataset)) if i not in index]
    else:
        train_data_list = [i for i in range(len(train_loader.dataset))]
    newbatches_x = np.zeros((10, train_loader.batch_size, 16, 1, 56, 56))
    newbatches_x1 = np.zeros((10, train_loader.batch_size, 16, 1, 56, 56))
    newbatches_l = np.zeros((10, train_loader.batch_size, 2))

    newbatches_x = torch.tensor(newbatches_x)
    newbatches_x1 = torch.tensor(newbatches_x1)
    newbatches_l = torch.tensor(newbatches_l)

    for i in range(train_loader.batch_size):
        random.shuffle(train_data_list)
        train_data = np.array(train_loader.dataset, dtype=object)[train_data_list]
        batches_x = []
        batches_x1 = []
        batches_l = []
        count = [[] for i in range(2)]

        for tempD in train_data:
            if tempD[2] == 0 and len(count[0]) < 5:
                batches_l.append([1, 0])
                batches_x.append(tempD[0][:, None, :, :])
                batches_x1.append(tempD[1][:, None, :, :])
                count[0].append(0)
            elif tempD[2] == 1 and len(count[1]) < 5:
                batches_l.append([0, 1])
                batches_x.append(tempD[0][:, None, :, :])
                batches_x1.append(tempD[1][:, None, :, :])
                count[1].append(1)
            elif len(batches_l) >= 10:
                break

        batches_x = [torch.from_numpy(batch_xi) for batch_xi in batches_x]
        batches_x1 = [torch.from_numpy(batch_xi) for batch_xi in batches_x1]
        batches_l = [torch.from_numpy(np.array(label_yi)) for label_yi in batches_l]

        batches_x = torch.stack(batches_x, 0)
        batches_x1 = torch.stack(batches_x1, 0)
        batches_l = torch.stack(batches_l, 0)
        # print('batches_x_size:', batches_x.shape)
        newbatches_x[:, i] = batches_x
        newbatches_x1[:, i] = batches_x1
        newbatches_l[:, i] = batches_l
    return_arr = cast_cuda([newbatches_x, newbatches_x1, newbatches_l.clone().detach()])

    return return_arr


def get_cos_sim(data):

    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean')
    cluster_l = cluster.fit_predict(data)

    return cluster_l


def get_psp_and_ttp(train_loader):

    index = train_loader.sampler.indices
    psp_index = []
    ttp_index = []

    for i in index:
        if train_loader.dataset[i][2] == 0:
            psp_index.append(i)
        else:
            ttp_index.append(i)

    psp_data = [train_loader.dataset[j] for j in psp_index]
    psp0_data = [psp[0] for psp in psp_data]
    psp0_data = np.array(psp0_data).reshape(len(psp_data), -1)
    psp_l = get_cos_sim(psp0_data)

    ttp_data = [train_loader.dataset[j] for j in ttp_index]
    ttp0_data = [ttp[0] for ttp in ttp_data]
    ttp0_data = np.array(ttp0_data).reshape(len(ttp_data), -1)
    ttp_l = get_cos_sim(ttp0_data)

    return psp_data, ttp_data, psp_l, ttp_l


def get_batch_data(train_loader, batchsize):

    index = range(len(train_loader.dataset))
    psp_index = []
    ttp_index = []

    for i in index:
        if train_loader.dataset[i][2]==0:
            psp_index.append(i)
        else:
            ttp_index.append(i)

    psp_n = random.choice([2,3])
    psp_d = random.sample(psp_index, psp_n)
    ttp_d = random.sample(ttp_index, batchsize-psp_n)
    batch_l = psp_d+ttp_d
    batch_d = [train_loader.dataset[j] for j in batch_l]
    random.shuffle(batch_d)
    data = [dx[0] for dx in batch_d]
    data1= [dx[1] for dx in batch_d]
    label= [dx[2] for dx in batch_d]

    return data, data1, label, batch_l


def train(model, optimizer, test_loader, train_loader, save_path, irun, fold):

    model.train()
    train_loss = 0.
    batch = 1
    TP = [0.]
    TN = [0.]
    FP = [0.]
    FN = [0.]
    ALL = 0.
    bankbatches = []
    bankbatch = []
    test_acc = 0

    for batch_idx in range(400):

        data, data1, label, batch_l = get_batch_data(train_loader, batchsize=5)
        [batches_x, batches_x1, batches_y] = get_task_batch(batch_l, train_loader)
        data = torch.tensor(data).unsqueeze(2)
        data1 = torch.tensor(data1).unsqueeze(2)
        target = torch.tensor(label, dtype=torch.long)

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            data1 = data1.cuda()
        if batch_idx % batch == 0:
            optimizer.zero_grad()

        Y_prob, l, batches, batchs = model([data, data1, target, batches_x, batches_x1, batches_y, True])
        loss = model.cross_entropy_loss(Y_prob, target)
        model.calculate_classification_error(Y_prob, target, TP, TN, FP, FN)
        model.update_banklist(batches, bankbatches)
        ALL += 1
        train_loss += loss

        if batch_idx % batch == 0:
            loss.backward()
            optimizer.step()
        if batch_idx % 5 == 0:
            model.eval()
            train_loss /= ALL
            ts_Accuracy, ts_Precision, ts_Recall, ts_F1, ts_rd, TPR, TNR, AUC = test(model, test_loader, train_loader,
                                                                                     bankbatches)
            if ts_Accuracy is not None and ts_Accuracy >= test_acc:
                test_acc = ts_Accuracy
                torch.save(model, save_path + 'am_new9_' + str(irun) + str(fold) + '_best_model.pkl')
                tts_Accuracy, tts_Precision, tts_Recall, tts_F1, tts_rd, tTPR, tTNR, tAUC \
                    = ts_Accuracy, ts_Precision, ts_Recall, ts_F1, ts_rd, TPR, TNR, AUC

            model.train()
    train_loss /= ALL
    Accuracy = (TP[0] + TN[0]) / (ALL*5)
    Precision = TP[0] / (TP[0] + FP[0]) if (TP[0] + FP[0]) != 0. else TP[0]
    Recall = TP[0] / (TP[0] + FN[0]) if (TP[0] + FN[0]) != 0. else TP[0]
    F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) != 0 else 2 * (Recall * Precision)

    return train_loss, Accuracy, Precision, Recall, F1, bankbatches, bankbatch, tts_Accuracy, tts_Precision, tts_Recall, tts_F1, tts_rd, tTPR, tTNR, tAUC


def test(model, test_loader, train_loader, bankbatches):

    model.eval()
    TP = [0.]
    TN = [0.]
    FP = [0.]
    FN = [0.]
    ALL = 0.
    outputs = []
    labels = []

    for batch_idx in range(8):

        data, data1, label, batch_l = get_batch_data(test_loader, batchsize=5)
        [batches_x, batches_x1, batches_y] = get_task_batch([], train_loader)
        data = torch.tensor(data).unsqueeze(2)
        data1 = torch.tensor(data1).unsqueeze(2)
        target = torch.tensor(label, dtype=torch.long)

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            data1 = data1.cuda()
        with torch.no_grad():
            Y_prob, _, _, batch = model([data, target, batches_x, batches_y, bankbatches])

        model.calculate_classification_error(Y_prob, target, TP, TN, FP, FN)
        ALL += 1
        outputs.append(Y_prob)
        labels.append(label)
    outputs = torch.stack(outputs).view(-1, 2).detach().cpu().numpy()
    labels = np.array(labels).reshape(-1)

    Accuracy = (TP[0] + TN[0]) / (ALL*5)
    Precision = TP[0] / (TP[0] + FP[0]) if (TP[0] + FP[0]) != 0. else TP[0]
    Recall = TP[0] / (TP[0] + FN[0]) if (TP[0] + FN[0]) != 0. else TP[0]
    F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) != 0 else 2 * (Recall * Precision)
    TPR = Recall
    TNR = TN[0] / (FP[0] + TN[0]) if (FP[0] + TN[0]) != 0. else TN[0]
    pre_scores = outputs[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(labels, pre_scores, pos_label=1)
    AUC = metrics.auc(fpr, tpr)

    return Accuracy, Precision, Recall, F1, [outputs, labels], TPR, TNR, AUC


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GNN')
    parser.add_argument('--metric_network', type=str, default='gnn', metavar='N',
                        help='gnn')
    parser.add_argument('--dataset', type=str, default='AD', metavar='N',
                        help='AD')
    parser.add_argument('--test_N_way', type=int, default=2, metavar='N')
    parser.add_argument('--train_N_way', type=int, default=2, metavar='N')
    parser.add_argument('--test_N_shots', type=int, default=5, metavar='N')
    parser.add_argument('--train_N_shots', type=int, default=5, metavar='N')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--feature_num', type=int, default=128, metavar='N',
                        help='feature number of one sample')
    parser.add_argument('--w_feature_num', type=int, default=128, metavar='N',
                        help='feature number for w computation')
    parser.add_argument('--w_feature_list', type=int, default=5, metavar='N',
                        help='feature list for w computation')
    parser.add_argument('--iterations', type=int, default=400, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
                        help='Decreasing the learning rate every x iterations')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', type=int, default=5, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--batch_size_test', type=int, default=5, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--batch_size_train', type=int, default=5, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                        help='Number of shots when training')
    parser.add_argument('--test_interval', type=int, default=40, metavar='N',
                        help='how many batches between each test')
    parser.add_argument('--random_seed', type=int, default=2023, metavar='N')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.feature_num = 64
    args.w_feature_num = 50
    timedata = time.strftime("%F")
    name = timedata + '-2-classe-fsl'
    save_path = 'result//{}//'.format(name)

    if name not in os.listdir('result//'):
        os.makedirs(save_path)
    torch.manual_seed(1)
    if os.path.isfile('all_cont_dataset3.pkl'):
        dataset = pickle.load(open('all_cont_dataset3.pkl', 'rb'))
    else:
        print("please select data sets!")
        sys.exit(0)

    run = 10
    ifolds = 4
    acc = np.zeros((run, ifolds), dtype=float)
    precision = np.zeros((run, ifolds), dtype=float)
    recall = np.zeros((run, ifolds), dtype=float)
    f_score = np.zeros((run, ifolds), dtype=float)
    auc = np.zeros((run, ifolds), dtype=float)
    tpr = np.zeros((run, ifolds), dtype=float)
    tnr = np.zeros((run, ifolds), dtype=float)
    index = 0
    ts_d = []

    for train_datasets, test_datasets in dataset:

        train_loader = torch.utils.data.DataLoader(
            train_datasets, batch_size=5, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test_datasets, batch_size=5, shuffle=True, drop_last=True)
        model = AMGraphBased28x28x1(args).cuda()
        optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.999), weight_decay=1e-3)
        test_acc = 0
        tempbankbatches = []
        irun, fold = divmod(index, 4)

        for epoch in range(0, 1):
            train_loss, tr_Accuracy, tr_Precision, tr_Recall, tr_F1, bankbatches, bankbatch, \
            ts_Accuracy, ts_Precision, ts_Recall, ts_F1, ts_rd, TPR, TNR, AUC = train(model, optimizer, test_loader,train_loader, save_path, irun, fold)

