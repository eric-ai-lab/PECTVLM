(
i=0;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-3 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=1;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-3 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=2;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=3;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-4 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=4;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-5 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=5;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-5 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=6;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-6 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=7;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language zh \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-6 \
--seed 1 \
--strategy method3 \
--text_loss_strategy contrastive \
--text_loss_ratio 1.0 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait