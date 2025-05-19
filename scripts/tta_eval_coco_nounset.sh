SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
EPOCH=$4
SUFFIX=exp
WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt
COCO_OUT_PATH=checkpoints/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER

COCO_LOG_FILE="$LOG_FOLDER/COCO_${TIME_START}.log"

python scene_policy_srs.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model ./pretrained/gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--threshold 0.2 \
--name_of_datasets coco \
--batch_size 1 \
--path_of_val_datasets ./annotations/coco/test_captions.json \
--name_of_entities_text coco_entities \
--image_folder ./annotations/coco/val2014/ \
--prompt_ensemble \
--weight_path=$WEIGHT_PATH \
--out_path=$COCO_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
--normalize_prefix \
--tta_steps 4 \
--tta_lr 5e-6 \
--tta_weight_decay 0.0 \
--reward_arch ViT-L/14 \
--reward_amplify 0 \
--reward_process 1 \
--process_batch 0 \
--momentum_update 0 \
--update_freq 64 \
--tta_momentum 0.9998 \
--update_w 1.0 \
--sample_k 4 \
--multiple_reward_models 0 \
--suffix ${SUFFIX}


$OTHER_ARGS \
|& tee -a  ${COCO_LOG_FILE}

echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $COCO_OUT_PATH/coco_generated_captions_${SUFFIX}.json |& tee -a  ${COCO_LOG_FILE}