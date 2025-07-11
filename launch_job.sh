#!/bin/bash

NAME="structa_DiT"
GPUS="4"
DIR="/users/rosenbaum/drosenba/code/DiT/dlc"
echo "#!/bin/bash" > /tmp/job.sh
echo "#SBATCH -J $NAME" >> /tmp/job.sh
echo "#SBATCH -o %x.%j.out" >> /tmp/job.sh
echo "#SBATCH -D $DIR" >> /tmp/job.sh
echo "#SBATCH -e %x.%j.err" >> /tmp/job.sh
echo "#SBATCH --time=50:00:00" >> /tmp/job.sh
echo "#SBATCH -G $GPUS" >> /tmp/job.sh
echo "#SBATCH --get-user-env" >> /tmp/job.sh
echo "#SBATCH --nodes 1" >> /tmp/job.sh
#echo "#SBATCH --gres=gpu:1" >> /tmp/job.sh
#echo "#SBATCH --cpus-per-task=1" >> /tmp/job.sh
#echo "#SBATCH --mem=4000MB" >> /tmp/job.sh


echo "srun --gpus=$GPUS --container-image /users/rosenbaum/drosenba/containers/dit.sqsh \
/bin/bash -c \"cd /root/code/DiT && torchrun --nnodes=1 --rdzv_endpoint=dgx06:12711 train_latent_tokens.py --data-path=/root/code/DiT/azmi_tokens/training_tokens.npz --normalization=1.0 --num-workers=$GPUS ${@:1} --model=DiT-XL/8 --global-batch-size=128 --expname=azmi_tokens  \"" >> /tmp/job.sh

#echo "srun --gpus=$GPUS --container-image /users/rosenbaum/drosenba/containers/dit.sqsh \
#/bin/bash -c \"cd /root/code/DiT && torchrun --nnodes=1 train_latent_tokens.py --data-path=/root/results/structured_functa/celeba_new/new_ad_tokens.npz --normalization=2.0 --num-workers=$GPUS ${@:1} --model=DiT-B/2 --expname=ad_tokens \"" >> /tmp/job.sh

#echo "srun --gpus=$GPUS --container-image /users/rosenbaum/drosenba/containers/dit.sqsh \
#/bin/bash -c \"cd /root/code/DiT && torchrun --nnodes=1 train_latent_tokens.py --data-path=/root/results/structured_functa/celeba_new/new_meta_tokens.npz --normalization=30.0 --num-workers=$GPUS ${@:1} --model=DiT-B/2 \"" >> /tmp/job.sh

# old runs that worked
#echo "srun --gpus=4 --container-image /users/rosenbaum/drosenba/containers/dit.sqsh \
#/bin/bash -c \"cd /root/code/DiT && torchrun --nnodes=1 train_latent_tokens.py --data-path=/root/results/structured_functa/celeba/celeba_latent_tokens_train.npz ${@:1} \"" >> /tmp/job.sh

# for clevr
#--data-path=/root/results/structured_functa/clevr/clevr_latent_tokens_train.npz ${@:1} \"" >> /tmp/job.sh

SUBMIT=`sbatch /tmp/job.sh`
echo $SUBMIT
JOB=`echo $SUBMIT | sed 's/.*job //'`
sleep 10
echo "Tailing $DIR/$NAME.$JOB.err:"
tail -f $DIR/$NAME.$JOB.err
