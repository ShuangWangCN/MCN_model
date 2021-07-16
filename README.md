# MCN_model

## dependencies
numpy == 1.18.5 <br>
pytorch == 1.6.0 <br>
PyG (torch-geometric) == 1.16.1 <br>
rdkit == 2020.03.3 <br>

## data preparation
The original data of protein and molecules are obtained from [DUD-E](http://dude.docking.org/). <br>

## train (cross validation)

The model is trained on The training dataset.<br>
**python train_MCN_target.py** <br>
## test
This is to do the prediction with the models we trained. And this step is to reproduce the experiments. <br>
**python Model_test.py** <br>



