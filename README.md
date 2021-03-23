# MCN_model

## dependencies
numpy == 1.18.5 <br>
pytorch == 1.6.0 <br>
PyG (torch-geometric) == 1.16.1 <br>
rdkit == 2020.03.3 <br>

## data preparation
1.The original data of protein and molecules are obtained from [DUD-E](http://dude.docking.org/). <br>
2.You can also download the prepared [datasets](https://drive.google.com/file/d/1ucKRh04-Uckj7oXi-kkPJ_qIYkYPkIbC/view?usp=sharing) which are suitable for the model:  <br>
(1) download the dataset.zip and unzip it. <br>
(2) copy the folders to your repo. <br>


## train (cross validation)
1.5-fold cross validation. <br>
**python 5_fold_train_MCN_target.py -f 1** <br>
where the parameters is fold number (1,2,3,4,5).<br>
2.training<br>
When the optimal parameters are selected by 5 folds cross validation, the model is trained on all training dataset.<br>
**python train_MCN_target.py** <br>
## test
This is to do the prediction with the models we trained. And this step is to reproduce the experiments. <br>
**python Model_test.py** <br>



