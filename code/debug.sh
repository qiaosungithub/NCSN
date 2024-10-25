source config.sh
rm -rf tmp # Comment this line if you want to reload (usually not the case)

CONDA_PATH=$(which conda)
CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
LOGDIR=$(pwd)/tmp
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

# hyperparameters
batch=1024
lr=0.0001
ep=5
wd=0.0

CONFIG=MNIST
model=NCSNv2
# CONFIG=tpu
source $CONDA_INIT_SH_PATH
export JAX_PLATFORMS=cpu

# remember to use your own conda environment
conda activate $OWN_CONDA_ENV_NAME

echo "start running main"

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root=${MNIST} \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=64 \
    --config.log_per_step=20 \
    --config.model=${model} \
    --config.optimizer='adamw' \
    --config.weight_decay=${wd} 

# note that the end of the final row should not contain a '\', or it will cause an error "Too many command-line arguments"