import time
import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import dgl
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import ConcatFeaturizer
from rdkit import Chem
from dgllife.utils import atom_type_one_hot, atom_degree_one_hot, atom_explicit_valence_one_hot, \
    atom_num_radical_electrons_one_hot, atom_is_aromatic_one_hot, atom_is_in_ring_one_hot, atom_total_num_H_one_hot, BaseAtomFeaturizer, \
    BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring, SMILESToBigraph

def GaussianSmearing(x):
    offset = np.linspace(-40.0,40.0,64)
    coeff = -0.5 / (offset[1] - offset[0])**2
    x = x.reshape(-1,1) - offset.reshape(1,-1)
    return np.exp(coeff * np.power(x, 2))


class DrugSynergyDataset():
    def __init__(self, dataset='loewe'):
        if dataset == 'loewe':
            self.data = pd.read_csv('../rawdata/oneil_loewe_cutoff30.txt', sep='\t')
        elif dataset == 'bliss':
            self.data = pd.read_csv('../rawdata/oneil_synergy_bliss.txt', sep='\t')
        elif dataset == 'hsa':
            self.data = pd.read_csv('../rawdata/oneil_synergy_hsa.txt', sep='\t')
        elif dataset == 'zip':
            self.data = pd.read_csv('../rawdata/oneil_synergy_zip.txt', sep='\t')
        self.dataset = dataset
        self.drugslist = sorted(list(set(list(self.data['drugname1']) + list(self.data['drugname2'])))) #38
        self.drugscount = len(self.drugslist)
        self.cellslist = sorted(list(set(self.data['cell_line']))) 
        self.cellscount = len(self.cellslist)

        self.drug_feat = pd.read_csv('../rawdata/oneil_drug_informax_feat.txt',sep='\t', header=None)
        self.drug_feat = torch.Tensor(np.array(self.drug_feat))
        self.cell_feat = np.load('../rawdata/oneil_cell_feat.npy')
        smiles_list = []
        with open('../rawdata/oneil_drug_smiles.txt', 'r') as f:
            fn = f.readlines()
            for line in fn:
                smiles_list.append(line.split('\t')[1].strip())
        self.drug_smi = smiles_list
        atom_concat_featurizer = ConcatFeaturizer([atom_type_one_hot, atom_degree_one_hot, atom_explicit_valence_one_hot, atom_num_radical_electrons_one_hot, atom_is_aromatic_one_hot, atom_is_in_ring_one_hot, atom_total_num_H_one_hot])
        self.smi_glist = []
        mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})
        bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])
        bond_featurizer = BaseBondFeaturizer({'type': bond_concat_featurizer}, self_loop=True)
        smi_to_g = SMILESToBigraph(add_self_loop=True, node_featurizer=mol_atom_featurizer,edge_featurizer=bond_featurizer)
        self.smi_g_num = []
        for smi in smiles_list:
            g = smi_to_g(smi)
            self.smi_glist.append(g)
            self.smi_g_num.append(g.num_nodes())
        self.smi_glist = dgl.batch(self.smi_glist)


    def get_feat(self):
        return self.drug_feat, self.cell_feat, self.drugslist, self.drugscount, self.cellscount

    def get_smi_graph(self):
        return self.smi_glist

    def get_graph(self, test_fold):
        valid_fold = list(range(10))[test_fold-1]
        train_fold = [ x for x in list(range(10)) if x != test_fold and x != valid_fold ]

        train_g_list = []
        valid_g_list = []
        test_g_list = []
        train_e_type = []
        valid_e_type = []
        test_e_type = []

        emb_g_list = [[],[],[]]

        if self.dataset == 'loewe':
            upb = 30
            lowb = 0
        elif self.dataset == 'bliss':
            upb = 3.68
            lowb = -3.37
        elif self.dataset == 'hsa':
            upb = 3.87
            lowb = -3.02
        elif self.dataset == 'zip':
            upb = 2.64
            lowb = -4.48
    
        for cellidx in range(self.cellscount):
            # cellidx = 0
            cellname = self.cellslist[cellidx]
            print('processing ', cellname)
            each_data = self.data[self.data['cell_line']==cellname]
            edges_src = [[],[],[]]
            edges_dst = [[],[],[]]
            edge_val = [[],[],[]]
            edge_type = [[],[],[]]

            edge_d_src = [[],[],[]]
            edge_d_dst = [[],[],[]]
            edge_d_val = [[],[],[]]

            for each in each_data.values:
                drugname1, drugname2, cell_line, synergy, fold = each
                drugidx1 = self.drugslist.index(drugname1)
                drugidx2 = self.drugslist.index(drugname2)

                if float(synergy) >= upb: #syn
                    if fold in train_fold:
                        edges_src[0].append(drugidx1)
                        edges_dst[0].append(drugidx2)
                        edges_src[0].append(drugidx2)
                        edges_dst[0].append(drugidx1)
                        edge_val[0].append(synergy)
                        edge_val[0].append(synergy)
                        edge_type[0].append(0)
                        edge_type[0].append(0)

                        edge_d_src[0].append(drugidx1)
                        edge_d_dst[0].append(drugidx2)
                        edge_d_src[0].append(drugidx2)
                        edge_d_dst[0].append(drugidx1)
                        edge_d_val[0].append(synergy)
                        edge_d_val[0].append(synergy)
                    elif fold == valid_fold:
                        edges_src[1].append(drugidx1)
                        edges_dst[1].append(drugidx2)
                        edges_src[1].append(drugidx2)
                        edges_dst[1].append(drugidx1)
                        edge_val[1].append(synergy)
                        edge_val[1].append(synergy)
                        edge_type[1].append(0)
                        edge_type[1].append(0)
                    elif fold == test_fold:
                        edges_src[2].append(drugidx1)
                        edges_dst[2].append(drugidx2)
                        edges_src[2].append(drugidx2)
                        edges_dst[2].append(drugidx1)
                        edge_val[2].append(synergy)
                        edge_val[2].append(synergy)
                        edge_type[2].append(0)
                        edge_type[2].append(0)
                elif (float(synergy) < upb) and (float(synergy) > lowb): #add
                    if fold in train_fold:
                        edges_src[0].append(drugidx1)
                        edges_dst[0].append(drugidx2)
                        edges_src[0].append(drugidx2)
                        edges_dst[0].append(drugidx1)
                        edge_val[0].append(synergy)
                        edge_val[0].append(synergy)
                        edge_type[0].append(1)
                        edge_type[0].append(1)

                        edge_d_src[1].append(drugidx1)
                        edge_d_dst[1].append(drugidx2)
                        edge_d_src[1].append(drugidx2)
                        edge_d_dst[1].append(drugidx1)
                        edge_d_val[1].append(synergy)
                        edge_d_val[1].append(synergy)
                    elif fold == valid_fold:
                        edges_src[1].append(drugidx1)
                        edges_dst[1].append(drugidx2)
                        edges_src[1].append(drugidx2)
                        edges_dst[1].append(drugidx1)
                        edge_val[1].append(synergy)
                        edge_val[1].append(synergy)
                        edge_type[1].append(1)
                        edge_type[1].append(1)
                    elif fold == test_fold:
                        edges_src[2].append(drugidx1)
                        edges_dst[2].append(drugidx2)
                        edges_src[2].append(drugidx2)
                        edges_dst[2].append(drugidx1)
                        edge_val[2].append(synergy)
                        edge_val[2].append(synergy)
                        edge_type[2].append(1)
                        edge_type[2].append(1)
                else:#ant
                    if fold in train_fold:
                        edges_src[0].append(drugidx1)
                        edges_dst[0].append(drugidx2)
                        edges_src[0].append(drugidx2)
                        edges_dst[0].append(drugidx1)
                        edge_val[0].append(synergy)
                        edge_val[0].append(synergy)
                        edge_type[0].append(2)
                        edge_type[0].append(2)

                        edge_d_src[2].append(drugidx1)
                        edge_d_dst[2].append(drugidx2)
                        edge_d_src[2].append(drugidx2)
                        edge_d_dst[2].append(drugidx1)
                        edge_d_val[2].append(synergy)
                        edge_d_val[2].append(synergy)
                    elif fold == valid_fold:
                        edges_src[1].append(drugidx1)
                        edges_dst[1].append(drugidx2)
                        edges_src[1].append(drugidx2)
                        edges_dst[1].append(drugidx1)
                        edge_val[1].append(synergy)
                        edge_val[1].append(synergy)
                        edge_type[1].append(2)
                        edge_type[1].append(2)
                    elif fold == test_fold:
                        edges_src[2].append(drugidx1)
                        edges_dst[2].append(drugidx2)
                        edges_src[2].append(drugidx2)
                        edges_dst[2].append(drugidx1)
                        edge_val[2].append(synergy)
                        edge_val[2].append(synergy)
                        edge_type[2].append(2)
                        edge_type[2].append(2)

            train_g = dgl.graph((edges_src[0], edges_dst[0]), num_nodes=self.drugscount)
            train_g.edata['syn'] = torch.Tensor(edge_val[0])
            train_g.edata['efeat'] = GaussianSmearing(torch.Tensor(edge_val[0]))
            train_g_list.append(train_g)
            valid_g = dgl.graph((edges_src[1], edges_dst[1]), num_nodes=self.drugscount)
            valid_g.edata['syn'] = torch.Tensor(edge_val[1])
            valid_g.edata['efeat'] = GaussianSmearing(torch.Tensor(edge_val[1]))
            valid_g_list.append(valid_g)
            test_g = dgl.graph((edges_src[2], edges_dst[2]), num_nodes=self.drugscount)
            test_g.edata['syn'] = torch.Tensor(edge_val[2])
            test_g.edata['efeat'] = GaussianSmearing(torch.Tensor(edge_val[2]))
            test_g_list.append(test_g)

            syn_g = dgl.graph((edge_d_src[0], edge_d_dst[0]), num_nodes=self.drugscount)
            syn_g.edata['syn'] = torch.Tensor(edge_d_val[0])
            syn_g.edata['efeat'] = GaussianSmearing(torch.Tensor(edge_d_val[0]))
            emb_g_list[0].append(syn_g)

            add_g = dgl.graph((edge_d_src[0], edge_d_dst[0]), num_nodes=self.drugscount)
            add_g.edata['syn'] = torch.Tensor(edge_d_val[0])
            add_g.edata['efeat'] = GaussianSmearing(torch.Tensor(edge_d_val[0]))
            emb_g_list[1].append(add_g)

            ant_g = dgl.graph((edge_d_src[0], edge_d_dst[0]), num_nodes=self.drugscount)
            ant_g.edata['syn'] = torch.Tensor(edge_d_val[0])
            ant_g.edata['efeat'] = GaussianSmearing(torch.Tensor(edge_d_val[0]))
            emb_g_list[2].append(ant_g)

        return train_g_list, valid_g_list, test_g_list, emb_g_list
    
