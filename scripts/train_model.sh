cd ~/sawh;
CUDA_VISIBLE_DEVICES=$1 python run_training.py \
model/prop=param \
model/wf_rep=implicit \
model.wf_rep.tn_num_layers=0 \
model.prop.rank=$2;