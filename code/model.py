import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import RelGraphConv, GraphConv
import dgl
from dgl.utils import expand_as_pair
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from dgl.nn import PathEncoder,SpatialEncoder,DegreeEncoder,GraphormerLayer
from dgl import shortest_dist
from torch.nn.utils.rnn import pad_sequence

class CelllineEncoder(nn.Module):
    def __init__(self, cfeat, dfeat, heads):
        super(CelllineEncoder, self).__init__()
        self.enc1 = nn.ModuleList([nn.Linear(cfeat, dfeat * 32) for i in range(heads)])
        self.dfeat = dfeat
        self.heads = heads

    def forward(self, cell_feat, x):
        #x: [1, d_num, d_dim]
        res = []
        for i in range(self.heads):
            w1 = self.enc1[i](cell_feat).reshape(1, self.dfeat, 32).expand(x.shape[0], self.dfeat, 32)
            h1 = torch.bmm(x, w1)
            h_mat = torch.bmm(h1, h1.transpose(1,2))
            res.append(h_mat)
        res = torch.stack(res, 3)
        return res

class GfPredictor_fuse(nn.Module):
    def __init__(self, df1, df2, df3, cf1, cf2):
        super(GfPredictor_fuse, self).__init__()
        self.dn1 = nn.Sequential(
            nn.Linear(df1, df1*2),
            nn.LeakyReLU(),
            nn.Linear(df1*2, df1),
        )

        self.dn3 = nn.Sequential(
            nn.Linear(df3, df3*2),
            nn.LeakyReLU(),
            nn.Linear(df3*2, df3),
        )

        self.cn1 = nn.Sequential(
            nn.Linear(cf1, cf1 * 2),
            nn.LeakyReLU(),
            nn.Linear(cf1 * 2, cf1),
        )

        self.cn2 = nn.Sequential(
            nn.Linear(cf2, cf2 * 2),
            nn.LeakyReLU(),
            nn.Linear(cf2 * 2, cf2),
        )

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cf1 + cf2 + (df1 + df3) * 2, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def apply_edges(self, edges):
        d1f = edges.src['h13']
        drug1_feat11_vector = self.dn1( d1f )
        d2f = edges.dst['h13']
        drug2_feat11_vector = self.dn1( d2f )
        drug1_feat3_vector = self.dn3( edges.src['h3'] ) 
        drug2_feat3_vector = self.dn3( edges.dst['h3'] )
        cell_feat_vector1 = self.cn1(edges.src['cf1'])
        cell_feat_vector2 = self.cn2(edges.src['cf2'])

        feat = torch.cat([drug1_feat11_vector, drug1_feat3_vector,\
            drug2_feat11_vector, drug2_feat3_vector,\
            cell_feat_vector1, cell_feat_vector2], 1)
        
        out = self.fc(feat)
        out = out.reshape(-1)
        return {'score': out}

    def forward(self, g, df11, df12, df13, df21, df22, df23, df3, cf1, cf2):
        with g.local_scope():
            g.ndata['h11'] = df11
            g.ndata['h12'] = df12
            g.ndata['h13'] = df13
            g.ndata['h21'] = df21
            g.ndata['h22'] = df22
            g.ndata['h23'] = df23
            g.ndata['h3'] = df3
            g.ndata['cf1'] = cf1.reshape(1, -1).expand(g.num_nodes(), -1)
            g.ndata['cf2'] = cf2.reshape(1, -1).expand(g.num_nodes(), -1)
            g.apply_edges(self.apply_edges)

            return g.edata['score']

class Fuse_gf(nn.Module):
    def __init__(self, dfeat, efeat, cfeat, afeat, hidden, mhidden, att_hidden):
        super(Fuse_gf, self).__init__()
        self.efeat = efeat
        self.hidden = hidden
        self.mhidden = mhidden
        self.cfl = nn.Linear(cfeat, hidden)
        self.dfl = nn.Linear(dfeat, hidden)

        self.Pe1 = nn.ModuleList()
        self.Se1 = nn.ModuleList()

        for i in range(3):
            self.Pe1.append(PathEncoder(6, efeat, 1))
            self.Se1.append(SpatialEncoder(6, 1))
        self.Me1 = nn.Linear(mhidden, 512)

        self.De1 = DegreeEncoder(32, 128, direction='in')
        self.Ce1 = CelllineEncoder(cfeat, dfeat, 1)
        self.gfls1 = GraphormerLayer(feat_size = hidden + 128 + hidden, hidden_size = att_hidden, num_heads = 1, attn_bias_type='add', norm_first=True, dropout=0.0, attn_dropout=0.0)
        
        
        self.Pe2 = nn.ModuleList()
        self.Se2 = nn.ModuleList()

        for i in range(3):
            self.Pe2.append(PathEncoder(6, efeat, 1))
            self.Se2.append(SpatialEncoder(6, 1))
        self.Me2 = nn.Linear(mhidden, 512)

        self.Ce2 = CelllineEncoder(cfeat, dfeat, 1)
        self.gfls2 = GraphormerLayer(feat_size = hidden + 128 + hidden, hidden_size = att_hidden, num_heads = 1, attn_bias_type='add', norm_first=True, dropout=0.0, attn_dropout=0.0)
        

        self.Pe3 = nn.ModuleList()
        self.Se3 = nn.ModuleList()

        for i in range(3):
            self.Pe3.append(PathEncoder(6, efeat, 1))
            self.Se3.append(SpatialEncoder(6, 1))
        self.Me3 = nn.Linear(mhidden, 512)

        self.Ce3 = CelllineEncoder(cfeat, dfeat, 1)
        self.gfls3 = GraphormerLayer(feat_size = hidden + 128 + hidden, hidden_size = att_hidden, num_heads = 1, attn_bias_type='add', norm_first=True, dropout=0.0, attn_dropout=0.0)
        

        self.conv1 = GraphConv(afeat + 128, mhidden)
        self.conv2 = GraphConv(mhidden + 128, mhidden)
        self.conv3 = GraphConv(mhidden + 128, mhidden)
        self.readout = WeightedSumAndMax(mhidden)
        self.cl = nn.Linear(cfeat, 128)
        self.p2 = nn.Linear(hidden + 128 + hidden, 128)
        self.p3 = nn.Linear(hidden + 128 + hidden, 128)

        self.pred = GfPredictor_fuse(hidden + 128 + hidden, mhidden, dfeat, cfeat, hidden)

    def forward(self, syn_g, drug_feat, cell_feat, tg, deg_l, dist_l, path_l, mol_gl):
        # mol graph layer 1
        x = mol_gl.ndata['h']
        cf = self.cl(cell_feat)
        cf = cf.reshape(1, -1).expand(mol_gl.num_nodes(), -1)
        pad_x = torch.cat([x, cf], 1)
        h1 = self.conv1(mol_gl, pad_x)
        res1 = self.readout(mol_gl, h1)[:,:self.mhidden]

        # syn graph layer 1
        df = self.dfl(drug_feat)
        deg_emb = self.De1(deg_l[0].unsqueeze(0))
        d_num = drug_feat.shape[0]
        c_num = 1
        in_feat = df.reshape(1, d_num, -1)
        in_feat = torch.cat([in_feat, deg_emb], 2) # C * D * (f + 32)
        cfv = self.cfl(cell_feat)
        cf = cfv.reshape(c_num, 1, -1).expand(c_num, d_num, -1)
        in_feat = torch.cat([in_feat, cf], 2)
        #in_feat = in_feat.reshape(1, d_num, 4, -1).transpose(2, 3).reshape(1, d_num, -1)
        bias = self.Ce1(cell_feat, drug_feat.unsqueeze(0))
        for i in range(3):
            pe = self.Pe1[i](dist_l[i].unsqueeze(0), path_l[i].unsqueeze(0))
            se = self.Se1[i](dist_l[i].unsqueeze(0))
            bias += pe + se
        th1  = self.Me1(res1)
        me = torch.mm(th1, th1.transpose(0, 1)).unsqueeze(0).unsqueeze(-1)
        bias += me
        
        sh1 = self.gfls1(in_feat, bias)
        
        # mol graph layer 2
        pd = []
        for i in range(mol_gl.batch_size):
            pdv = self.p2(sh1[0][i])
            pd.append(pdv.expand(mol_gl.batch_num_nodes()[i], -1))
        pd = torch.cat(pd, 0)
        ph1 = torch.cat([h1, pd], 1)
        h2 = self.conv2(mol_gl, ph1)
        res2 = self.readout(mol_gl, h2)[:,:self.mhidden]

       
        bias = self.Ce2(cell_feat, drug_feat.unsqueeze(0))
        for i in range(3):
            pe = self.Pe2[i](dist_l[i].unsqueeze(0), path_l[i].unsqueeze(0))
            se = self.Se2[i](dist_l[i].unsqueeze(0))
            bias += pe + se
        th2  = self.Me2(res2)
        me = torch.mm(th2, th2.transpose(0, 1)).unsqueeze(0).unsqueeze(-1)
        bias += me
        
        sh2 = self.gfls2(sh1, bias)
        # mol graph layer 2
        pd = []
        for i in range(mol_gl.batch_size):
            pdv = self.p3(sh1[0][i])
            pd.append(pdv.expand(mol_gl.batch_num_nodes()[i], -1))
        pd = torch.cat(pd, 0)
        ph2 = torch.cat([h2, pd], 1)
        h3 = self.conv3(mol_gl, ph2)
        res3 = self.readout(mol_gl, h3)[:,:self.mhidden]

       
        bias = self.Ce3(cell_feat, drug_feat.unsqueeze(0))
        for i in range(3):
            pe = self.Pe3[i](dist_l[i].unsqueeze(0), path_l[i].unsqueeze(0))
            se = self.Se3[i](dist_l[i].unsqueeze(0))
            bias += pe + se
        th3  = self.Me3(res3)
        me = torch.mm(th3, th3.transpose(0, 1)).unsqueeze(0).unsqueeze(-1)
        bias += me
        
        sh3 = self.gfls3(sh2, bias)
        
        score = self.pred(tg, sh1.squeeze(0), sh2.squeeze(0), sh3.squeeze(0), res1, res2, res3, drug_feat, cell_feat, cfv)

        return score
