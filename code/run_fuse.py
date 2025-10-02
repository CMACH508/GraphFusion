from dataset import DrugSynergyDataset
from model import Fuse_gf
import numpy as np
import torch
import pandas as pd
import dgl
import os
import time
from datetime import datetime
import logging
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score
from tqdm import tqdm
from dgl import shortest_dist

def calc_stat(numbers):
    mu = sum(numbers) / len(numbers)
    sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
    return mu, sigma


def conf_inv(mu, sigma, n):
    delta = 2.776 * sigma / (n ** 0.5)  # 95%
    return mu - delta, mu + delta

data = DrugSynergyDataset()

test_res = []
test_label = []
test_losses = []
test_pccs = []
n_delimiter = 60
class_stats = np.zeros((10, 7))

log_file = 'cv.log'
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=logging.INFO)


for test_fold in range(10):
    tg, vg, eg, emb_gl = data.get_graph(test_fold)
    smi_g_list = data.get_smi_graph()
    drug_feat, cell_feat, drugslist, drugscount, cellscount = data.get_feat()
    cell_feat = torch.Tensor(cell_feat)
    model = Fuse_gf(drug_feat.shape[1], tg[0].edata['efeat'].shape[1], cell_feat.shape[1], smi_g_list.ndata['h'].shape[1], 256, 512, 1024)
    device = torch.device('cuda:{:d}'.format(1))
    torch.cuda.set_device(device)
    model.to(device)
    smi_g_list = smi_g_list.to(device)

    for i in range(cellscount):
        tg[i] = tg[i].to(device)
        vg[i] = vg[i].to(device)
        eg[i] = eg[i].to(device)
        emb_gl[0][i] = emb_gl[0][i].to(device)
        emb_gl[1][i] = emb_gl[1][i].to(device)
        emb_gl[2][i] = emb_gl[2][i].to(device)

    drug_feat = drug_feat.to(device)
    cell_feat = cell_feat.to(device)

    zpd = torch.zeros(1, tg[0].edata['efeat'].shape[1]).to(device)
    max_nodes = tg[0].num_nodes()

    dist_l = -torch.ones((3, len(tg), max_nodes, max_nodes), dtype=torch.long).to(device)
    path_l = torch.zeros([3, len(tg), max_nodes, max_nodes, 6, tg[0].edata['efeat'].shape[1]]).to(device)
    deg_l = [[],[],[]]

    for tp in range(3):
        for i in range(len(tg)):
            dist, path = shortest_dist(emb_gl[tp][i], root=None, return_paths=True)
            n_node = emb_gl[tp][i].num_nodes()
            dist_l[tp,i,:n_node, :n_node] = dist
            pde = torch.cat([emb_gl[tp][i].edata['efeat'], zpd], dim=0)
            md = dist.max()
            path_l[tp,i, :n_node, :n_node, :md, :] = pde[path[:,:,:6]]
            deg_l[tp].append(emb_gl[tp][i].in_degrees())
        deg_l[tp] = torch.stack(deg_l[tp], 0)

    deg_l = torch.stack(deg_l, 0)
    deg_l = deg_l.to(device)
    deg_l = deg_l.transpose(0, 1)
    dist_l = dist_l.transpose(0, 1)
    path_l = path_l.transpose(0, 1)

    lbd = 1
    opt = torch.optim.Adam([{'params':model.parameters()}], lr=1e-4)

    min_loss = 1e9
    min_epoch = 0
    loss_func = nn.MSELoss(reduction='sum')

    for epoch in range(10000):
        mse_l = 0
        tot_l = 0
        t_loss = 0
        model.train()

        for i in range(cellscount):
            loss = 0
            logits = model(tg[i], drug_feat, cell_feat[i], tg[i], deg_l[i], dist_l[i], path_l[i], smi_g_list)
            edge_label = tg[i].edata['syn']
            loss += loss_func(logits, edge_label)
            mse_l += loss_func(logits, edge_label).item()
            vx = logits - torch.mean(logits)
            vy = edge_label - torch.mean(edge_label)
            if (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) > 1e-7:
                loss += logits.shape[0] * 8000 * (1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
            tot_l += logits.shape[0]
            t_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        #mse_l = mse_l.item()

        pcc_l = t_loss - mse_l
        pcc_l = 1.0 * pcc_l / tot_l
        mse_l = 1.0 * mse_l / tot_l

        print('Epoch : {}, train loss : {}'.format(epoch, t_loss / tot_l))
        print('Epoch : {}, train mse loss : {}, pcc loss : {}'.format(epoch, mse_l, pcc_l))

        with torch.no_grad():
            model.eval()
            val_loss = 0
            mse_l = 0
            tot_l = 0

            for i in range(cellscount):
                logits = model(tg[i], drug_feat, cell_feat[i], vg[i], deg_l[i], dist_l[i], path_l[i], smi_g_list)
                edge_label = vg[i].edata['syn']
                val_loss += loss_func(logits, edge_label)
                mse_l += loss_func(logits, edge_label)
                vx = logits - torch.mean(logits)
                vy = edge_label - torch.mean(edge_label)
                if (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) > 1e-7:
                    val_loss += logits.shape[0] * 200 * (1 - torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
                tot_l += logits.shape[0]

            mse_l = mse_l.item()
            pcc_l = val_loss.item() - mse_l
            pcc_l = 1.0 * pcc_l / tot_l
            mse_l = 1.0 * mse_l / tot_l
            val_loss = val_loss.item()
            print('Epoch : {}, valid loss : {}'.format(epoch, val_loss))
            print('MSE : {}, Pcc : {}'.format(mse_l, 1 - pcc_l / 200))
            
            if (val_loss < min_loss):
                min_loss = val_loss
                min_epoch = epoch
                print('Current best loss : {}, Epoch : {}'.format(mse_l, min_epoch))
                torch.save(model.state_dict(), '../ckpt/best_model_{}.ckpt'.format(test_fold))

            if min_epoch + 1000 < epoch:
                break            
            
    model.load_state_dict(torch.load('../ckpt/best_model_{}.ckpt'.format(test_fold)))
    model.eval()

    with torch.no_grad():
        y_true = []
        y_pred = []

        mse_l = 0
        tot_l = 0

        for i in range(cellscount):
            logits = model(tg[i], drug_feat, cell_feat[i], eg[i], deg_l[i], dist_l[i], path_l[i], smi_g_list)
            edge_label = eg[i].edata['syn']
            vx = logits - torch.mean(logits)
            vy = edge_label - torch.mean(edge_label)
            y_true.append(edge_label)
            y_pred.append(logits)

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        test_loss = loss_func(y_pred, y_true).item()
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().flatten()
        test_pcc = np.corrcoef(y_pred, y_true)[0, 1]
        test_loss /= len(y_true)
        y_pred_binary = [ 1 if x >= 30 else 0 for x in y_pred ]
        y_true_binary = [ 1 if x >= 30 else 0 for x in y_true ]
        roc_score = 0
        try:
            roc_score = roc_auc_score(y_true_binary, y_pred)
        except ValueError:
            pass
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        auprc_score = auc(recall, precision)
        accuracy = accuracy_score( y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary)
        kappa = cohen_kappa_score(y_true_binary, y_pred_binary)

    class_stat = [roc_score, auprc_score, accuracy, f1, precision, recall, kappa]
    class_stats[test_fold] = class_stat
    test_losses.append(test_loss)
    test_pccs.append(test_pcc)
    logging.info("Test loss: {:.4f}".format(test_loss))
    logging.info("Test pcc: {:.4f}".format(test_pcc))
    logging.info("*" * n_delimiter + '\n')
    
     
logging.info("CV completed")
mu, sigma = calc_stat(test_losses)
logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
lo, hi = conf_inv(mu, sigma, len(test_losses))
logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
rmse_loss = [x ** 0.5 for x in test_losses]
mu, sigma = calc_stat(rmse_loss)
logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
pcc_mean, pcc_std = calc_stat(test_pccs)
logging.info("pcc: {:.4f} ± {:.4f}".format(pcc_mean, pcc_std))

class_stats = np.concatenate([class_stats, class_stats.mean(axis=0, keepdims=True), class_stats.std(axis=0, keepdims=True)], axis=0)
pd.DataFrame(class_stats).to_csv('class_stats.txt', sep='\t', header=None, index=None)

