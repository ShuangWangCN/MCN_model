import numpy as np
from MCN.utils import read_pkl,save_pkl
root_dir= '../dataset'
dir= '../dataset/pro_data/'
train_data=read_pkl(dir+'dict_index_target_train.pickle')
print('a')
protein_name_all=np.load(root_dir+'/pro_data/protein_name.npy')
protein_name_all=protein_name_all.tolist()
test_dude_gene = ['egfr', 'parp1', 'fnta', 'aa2ar', 'pygm', 'kith', 'met', 'abl1', 'ptn1', 'casp3', 'hdac8', 'grik1', 'kpcb', 'ada', 'pyrd', 'ace', 'aces', 'pgh1', 'aldr', 'kit', 'fa10', 'pa2ga', 'fgfr1', 'cp3a4', 'wee1', 'tgfr1']
train_in_pro=[]
for pro_item in protein_name_all:
    if pro_item not in test_dude_gene:
        train_in_pro.append(pro_item)
print(len(train_in_pro))#68 train proteins
# random.shuffle(train_in_pro)
fold_1=train_in_pro[:14]
fold_2=train_in_pro[14:28]
fold_3=train_in_pro[28:42]
fold_4=train_in_pro[42:55]
fold_5=train_in_pro[55:68]
fold_1_dict=[]
fold_2_dict=[]
fold_3_dict=[]
fold_4_dict=[]
fold_5_dict=[]
for train_dict_item in train_data:
    dict_item_target=train_dict_item['target']
    if dict_item_target in fold_1:
        fold_1_dict.append(train_dict_item)
    if dict_item_target in fold_2:
        fold_2_dict.append(train_dict_item)
    if dict_item_target in fold_3:
        fold_3_dict.append(train_dict_item)
    if dict_item_target in fold_4:
        fold_4_dict.append(train_dict_item)
    if dict_item_target in fold_5:
        fold_5_dict.append(train_dict_item)
print('number of fold 1:',len(fold_1_dict))
print('number of fold 2:',len(fold_2_dict))
print('number of fold 3:',len(fold_3_dict))
print('number of fold 4:',len(fold_4_dict))
print('number of fold 5:',len(fold_5_dict))
fold_1_valid=fold_1_dict
fold_1_train=fold_2_dict+fold_3_dict+fold_4_dict+fold_5_dict
print('number of fold 1 train:',len(fold_1_train))
fold_2_valid=fold_2_dict
fold_2_train=fold_1_dict+fold_3_dict+fold_4_dict+fold_5_dict
print('number of fold 2 train:',len(fold_2_train))
fold_3_valid=fold_3_dict
fold_3_train=fold_1_dict+fold_2_dict+fold_4_dict+fold_5_dict
print('number of fold 3 train:',len(fold_3_train))
fold_4_valid=fold_4_dict
fold_4_train=fold_1_dict+fold_2_dict+fold_3_dict+fold_5_dict
print('number of fold 4 train:',len(fold_4_train))

fold_5_valid=fold_5_dict
fold_5_train=fold_1_dict+fold_2_dict+fold_3_dict+fold_4_dict
print('number of fold 5 train:',len(fold_5_train))

save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_1_valid.pkl',fold_1_valid)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_2_valid.pkl', fold_2_valid)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_3_valid.pkl', fold_3_valid)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_4_valid.pkl', fold_4_valid)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_5_valid.pkl', fold_5_valid)


save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_1_train.pkl',fold_1_train)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_2_train.pkl', fold_2_train)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_3_train.pkl', fold_3_train)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_4_train.pkl', fold_4_train)
save_pkl('../dataset/pro_data/5_fold/dict_index_target_fold_5_train.pkl', fold_5_train)

print('end')