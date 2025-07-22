#!/bin/bash

NAME="npdit"
GPUS="2"
DIR="/users/rosenbaum/drosenba/code/token-DiT/dlc"
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
/bin/bash -c \"cd /root/code/token-DiT && torchrun --nnodes=1 --rdzv_endpoint=\$SLURM_NODELIST:12711 train_np_dit.py --data-path=/root/data/celeba/ --num-workers=$GPUS ${@:1} --model=DiT-B --global-batch-size=16 --expname=celeba_baseline_\$SLURM_JOB_ID  \"" >> /tmp/job.sh


SUBMIT=`sbatch /tmp/job.sh`
echo $SUBMIT
JOB=`echo $SUBMIT | sed 's/.*job //'`
sleep 10
echo "Tailing $DIR/$NAME.$JOB.err:"
tail -f $DIR/$NAME.$JOB.err
