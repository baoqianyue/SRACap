SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

DEVICE=$1
EXP_NAME=`echo "$(basename $0)" | cut -d'.' -f1`
EXP_NAME=$EXP_NAME+'exp'
LOG_FILE=logs/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}
LOG_FILE="$LOG_FOLDER/${TIME_START}.log"
mkdir -p $LOG_FOLDER

echo "=========================================================="
echo "RUNNING EXPERIMENTS: $EXP_NAME, saving in checkpoints/$EXP_NAME"
echo "=========================================================="

python main.py \
--bs 80 \
--lr 0.00002 \
--epochs 60 \
--device cuda:$DEVICE \
--random_mask \
--prob_of_random_mask 0.4 \
--clip_model ViT-B/32 \
--language_model gpt2 \
--using_hard_prompt \
--continuous_prompt_length 10 \
--soft_prompt_first \
--max_num_of_entities 5 \
--path_of_sg ./annotations/cc3m/cc3m_texts_sg_features_ViT-B32.pickle \
--num_workers 4 \
--out_dir checkpoints/$EXP_NAME \
|& tee -a  ${LOG_FILE}