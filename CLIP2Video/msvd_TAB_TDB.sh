echo `date "+%Y-%m-%d %H:%M:%S"`
VERSION=/share/home/lyq/GLAattack/CLIP2Video
DATA_PATH=${VERSION}/data/msvd_data/
OUTPUT_FILE=/share/home/lyq/GLAattack/CLIP2Video/CLIP2Video_MSVD
CHECKPOINT=/share/home/lyq/CLIP2Video/CLIP2Video_MSVD
MODEL_NUM=2

python ${VERSION}/infer_retrieval.py \
--num_thread_reader=4 \
--data_path ${DATA_PATH} \
--features_path /share/test/lyq/GLAattack/clip2video/adv/ViT-B-16#rebuttal_DA_ADAM_adam#cos-layer4_steps60 \
--output_dir ${OUTPUT_FILE}/ViT-B-16#rebuttal_DA_ADAM_adam#cos-layer4_steps60 \
--max_words 32 \
--max_frames 12 \
--batch_size_val 64 \
--datatype msvd \
--feature_framerate 1 \
--sim_type seqTransf \
--checkpoint ${CHECKPOINT} \
--do_eval \
--model_num ${MODEL_NUM} \
--temporal_type TDB \
--temporal_proj sigmoid_selfA \
--center_type TAB \
--centerK 5 \
--center_weight 0.5 \
--center_proj TAB_TDB \
--clip_path /share/home/lyq/.cache/clip/ViT-B-32.pt

#--myreplace /share/test/lyq/video/MSVD_dr_16_short 

echo `date "+%Y-%m-%d %H:%M:%S"`