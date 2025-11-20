cd ~/sawh;
data_path=./;
CUDA_VISIBLE_DEVICES=$1 python run_cgh.py \
cgh/supervision=sah \
cgh/lf_data=olas \
cgh.data_type=4d \
cgh.data_path=$data_path \
cgh.num_iters=1000 \
cgh.steered_angle="[[0.00,0.00]]" \
cgh.rank=1 \
model/prop=param \
model.model_path=$2 \
model/wf_rep=implicit \
model.wf_rep.tn_num_layers=0 \
model.prop.rank=$3;