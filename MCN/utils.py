from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch,pickle
import numpy as np
from MCN.molecule_prepare import smile_to_graph

def collate(data_list):
    mol_data_list=[data[0] for data in data_list]
    batchA = Batch.from_data_list(mol_data_list)
    site_3D_list=[data[1] for data in data_list]
    site_3D_list_tensor=torch.stack(site_3D_list).squeeze(dim=1)
    Seq_1D_list=[data[2].long() for data in data_list]
    Seq_1D_list_tensor=torch.stack(Seq_1D_list).squeeze(dim=1)
    return batchA, site_3D_list_tensor,Seq_1D_list_tensor
def get_pro_3D(dir,name):
    pro_3D=np.load(dir+name+'.npy')
    return pro_3D

def get_pro_seq(dir,name):
    pro_seq_file=open(dir,'rb')
    pro_seq=pickle.load(pro_seq_file)
    pro_seq_file.close()
    pro_seq=[item.get('pro_seq') for item in pro_seq if item.get('pro_name')==name ]
    seq_coding=seq_1D_process(pro_seq[0])
    return seq_coding

def seq_1D_process(seq_string):
    seq_char_list=list(seq_string)
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: i for i, v in enumerate(seq_voc)}
    max_seq_len = 1000
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(seq_char_list[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def pro_descriptor(dir,name):
    site_3D=get_pro_3D(dir+'/30/',name)
    seq_coding=get_pro_seq(dir+'/pro_seq.pkl',name)
    return site_3D,seq_coding

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()
def save_pkl(dir,file):
    dir_pkl=open(dir,'wb')
    pickle.dump(file,dir_pkl)
    dir_pkl.close()
def read_pkl(dir):
    dir_pkl=open(dir,'rb')
    pkl_file = pickle.load(dir_pkl)
    dir_pkl.close()
    return pkl_file


def check_chain(pro_line,acd_pos):
    if pro_line[acd_pos] == 1:
        acd_chain = pro_line[acd_pos]
        acd_No = pro_line[acd_pos+1]
    if pro_line[acd_pos] > 1:
        acd_chain = pro_line[acd_pos][0]
        acd_No = pro_line[acd_pos]
    return acd_chain,acd_No

def train(model, device, train_loader, optimizer, epoch,FocalLoss,batch_size):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    TRAIN_BATCH_SIZE = batch_size
    for batch_idx, batch in enumerate(train_loader):
        data_mol = batch[0].to(device)
        site_3D = batch[1].to(device)
        pro_1D = batch[2].to(device)
        optimizer.zero_grad()
        output = model(data_mol, site_3D, pro_1D)
        data_mol_tensor=data_mol.y.view(-1, 1).float().to(device)
        loss = FocalLoss(output, data_mol_tensor,device)

        loss.backward()
        optimizer.step()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx,batch in enumerate(loader):
            data_mol = batch[0].to(device)
            site_3D = batch[1].to(device)
            pro_1D = batch[2].to(device)
            output = model(data_mol, site_3D, pro_1D)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


class DTADataset(InMemoryDataset):
    def __init__(self, dir=None, data_list=None, transform=None,pre_transform=None):

        super(DTADataset, self).__init__( transform, pre_transform)
        self.dir=dir
        self.data_list=data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        dict_item=self.data_list[idx]
        pro_name = dict_item.get('target')
        ligand = dict_item.get('ligand')
        label = dict_item.get('label')
        site_3D, seq_coding = pro_descriptor(self.dir,pro_name)
        c_size, features, edge_index=smile_to_graph(ligand)
        GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([label]))
        GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
        GCNData_mol.__setitem__('smiles', ligand)
        GCNData_mol.__setitem__('pro', pro_name)


        site_3D_Features=torch.Tensor([site_3D])
        seq_coding_features=torch.Tensor([seq_coding])

        return GCNData_mol,site_3D_Features,seq_coding_features

