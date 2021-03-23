from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from MCN.utils import collate,train,predicting,read_pkl,DTADataset
from MCN.model_net import Model_Net
from MCN.Focal_loss import FocalLoss
from sklearn.metrics import roc_auc_score
from optparse import OptionParser

device=torch.device('cuda:1')
random_seed=100
torch.cuda.manual_seed(random_seed)
parser = OptionParser()
parser.add_option("-d", "--dir", dest="dir",default='dataset/pro_data/')
parser.add_option("-r", "--learning", dest="learning_rate",default=0.0002)
parser.add_option("-f", "--fold", dest="fold_number",default=1)
parser.add_option("-b", "--batch", dest="batch_size", default=64)
parser.add_option("-e", "--epoch", dest="epoch_size", default=30)
opts,args = parser.parse_args()
batch_size = int(opts.batch_size)
epoch_size=int(opts.epoch_size)
dir=opts.dir
fold_number=int(opts.fold_number)
learning_rate=float(opts.learning_rate)

train_data=read_pkl(dir+'5_fold/dict_index_target_fold_'+str(fold_number)+'_train.pkl')
valid_data=read_pkl(dir + '5_fold/dict_index_target_fold_' + str(fold_number) + '_valid.pkl')

train_data=DTADataset(dir=dir,data_list=train_data)
valid_data=DTADataset(dir=dir, data_list=valid_data)
model = Model_Net()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()
FocalLoss=FocalLoss(alpha=0.2, gamma=2)
setting_name='5_fold_number_('+str(fold_number)+')_target_splitting_batch'+str(batch_size)+'--epoch_'+str(epoch_size)
print(setting_name)

dataset_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16,
                           collate_fn=collate, drop_last=False)
dataset_valid = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=16,
                           collate_fn=collate, drop_last=False)

for epoch in range(epoch_size):
    train(model, device, dataset_train, optimizer, epoch + 1,FocalLoss,batch_size)
    scheduler.step()
torch.save(model.state_dict(), "models/" + setting_name+'.pkl')
print('predicting for valid data')
G_valid, P_valid = predicting(model, device, dataset_valid)
valid_roc = roc_auc_score(G_valid, P_valid)
print('predicting valid AUC:',valid_roc)









