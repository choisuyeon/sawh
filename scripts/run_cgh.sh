cd ~/sawh;
data_path=./;
CUDA_VISIBLE_DEVICES=$1 python run_cgh.py \
cgh/supervision=sah \
cgh/lf_data=olas \
cgh.data_type=4d \
cgh.data_path=$data_path \
cgh.num_iters=1000 \
cgh.rank=$2 \
