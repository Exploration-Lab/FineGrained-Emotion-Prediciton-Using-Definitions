# FineGrained-Emotion-Prediciton-Using-Definitions

pip install -r requirements.txt

## Class definition prediction(CDP)<br/>
python run_def_nsp.py --taxonomy 'config file name'

## Masked language modelling(MLM)<br/>
python run_def_mlm_posonly.py --taxonomy 'config file name'
#### For all examples:<br/>
python run_def_mlm.py --taxonomy 'config file name'

## Combined setup(CDP+MLM)<br/>
python run_def_nsp-mlm.py --taxonomy 'config file name'

## Reference
https://github.com/monologg/GoEmotions-pytorch<br/>
https://huggingface.co/
