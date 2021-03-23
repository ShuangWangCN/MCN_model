from torch.utils.data import DataLoader
import torch
from MCN.utils import collate,predicting,read_pkl,DTADataset
from MCN.model_net import Model_Net
from sklearn.metrics import roc_auc_score
from optparse import OptionParser
device=torch.device('cuda:1')

random_seed=100
torch.cuda.manual_seed(random_seed)
parser = OptionParser()
parser.add_option("-d", "--dir", dest="dir",default='dataset/pro_data')
parser.add_option("-r", "--learning", dest="learning_rate",default=0.0002)
parser.add_option("-b", "--batch", dest="batch_size", default=64)
parser.add_option("-e", "--epoch", dest="epoch_size", default=30)
opts,args = parser.parse_args()
batch_size = int(opts.batch_size)
epoch_size=int(opts.epoch_size)
dir=opts.dir
file_name='final_model_0.976'
model_path='models/'+file_name+'.pkl'
test_data=read_pkl(dir+'/dict_index_target_test.pickle')
model = Model_Net( )
model = model.to(device)

if model_path is not None:
    model.load_state_dict(torch.load(model_path))

test_data=DTADataset(dir=dir,data_list=test_data)
dataset_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16,
                          collate_fn=collate, drop_last=False)

G, P = predicting(model, device, dataset_test)
test_roc = roc_auc_score(G, P)
print(test_roc)



