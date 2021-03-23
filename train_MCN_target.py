# this file is designed for the dataset splitted by protein target
# only train for train  and test to choose a model
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
parser.add_option("-b", "--batch", dest="batch_size", default=64)
parser.add_option("-e", "--epoch", dest="epoch_size", default=30)
opts,args = parser.parse_args()
batch_size = int(opts.batch_size)
epoch_size=int(opts.epoch_size)
dir=opts.dir

learning_rate=float(opts.learning_rate)
train_data=read_pkl(dir+'dict_index_target_train.pickle')
test_data=read_pkl(dir+'dict_index_target_test.pickle')
train_data=DTADataset(dir=dir,data_list=train_data)
test_data=DTADataset(dir=dir,data_list=test_data)

model = Model_Net( )
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()
FocalLoss=FocalLoss( alpha=0.2, gamma=2)
epoch_result=[]
dataset_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16,
                           collate_fn=collate, drop_last=False)
dataset_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16,
                          collate_fn=collate, drop_last=False)
for epoch in range(epoch_size):
    test_loss_epoch=0
    train(model, device, dataset_train, optimizer, epoch + 1,FocalLoss,batch_size)
    scheduler.step()
torch.save(model.state_dict(), 'models/model.pkl')
print('predicting for test data')
G, P = predicting(model, device, dataset_test)
test_roc = roc_auc_score(G, P)
print('predicting test AUC:',test_roc)








