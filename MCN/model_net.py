import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool as gep


class Model_Net(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xt=25,embed_dim=128,num_features_mol=73, output_dim=128, dropout=0.2):
        super(Model_Net, self).__init__()
        self.n_output = n_output
        #mol network
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, 2 * output_dim)
        #pro  network
        self.pro_conv1 = self.pro_3D_conv(8, 32)
        self.pro_conv2 = self.pro_3D_conv(32, 64)
        self.pro_conv3 = self.pro_3D_conv(64, 128)
        self.pro_relu = nn.LeakyReLU()
        self.pro_fc1 = nn.Linear(1024, 128)
        self.pro_fc2 = nn.Linear(128, output_dim)
        #sequence  network
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.seq_conv1 = self.pro_seq_conv(embed_dim, int(0.5*embed_dim) )
        self.seq_conv2 = self.pro_seq_conv(int(0.5*embed_dim), int(0.25*embed_dim))
        self.seq_conv3 = self.pro_seq_conv(int(0.25*embed_dim), int(0.25*embed_dim))
        self.seq_fc1 = nn.Linear(4032, 2*output_dim)
        self.seq_fc2 = nn.Linear( 2*output_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # combined layers
        self.fc1 = nn.Linear(4* output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def pro_3D_conv(self,pro_in,pro_out):
        conv_layer = nn.Sequential(
            nn.Conv3d(pro_in, pro_out, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2),stride=(2,2,2)),
        )
        return conv_layer
    def pro_seq_conv(self,seq_in,seq_out):
        conv_1D_layer=nn.Sequential(nn.Conv1d(seq_in,seq_out,kernel_size=5,padding=3),
                                    nn.LeakyReLU(),
                                    nn.MaxPool1d(kernel_size=2,stride=2),)
        return conv_1D_layer


    def forward(self, data_mol, data_site_3D,data_seq_1D):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        batch_site_3D_8_channels = data_site_3D
        batch_seq_1D = data_seq_1D
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        #3D pro_process
        batch_site_3D_8_channels=batch_site_3D_8_channels.permute(0,4,1,2,3)
        x_pro = self.pro_conv1(batch_site_3D_8_channels)
        x_pro = self.pro_conv2(x_pro)
        x_pro = self.pro_conv3(x_pro)
        x_pro=x_pro.view(-1,1024)
        x_pro = self.pro_fc1(x_pro)
        x_pro = self.pro_fc2(x_pro)

        #1D seq_process
        seq_embedding=self.embedding_xt(batch_seq_1D)
        seq_embedding= seq_embedding.permute(0,2,1)
        x_seq = self.seq_conv1(seq_embedding)
        x_seq = self.seq_conv2(x_seq)
        x_seq = self.seq_conv3(x_seq)
        x_seq=x_seq.view(x_seq.size()[0],-1)
        x_seq = self.seq_fc1(x_seq)
        x_seq = self.seq_fc2(x_seq)

        xc = torch.cat((x, x_pro, x_seq), 1)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
