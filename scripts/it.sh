(
i=5;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language it \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.5 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=6;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language it \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-4 \
--seed 1 \
--strategy method2 \
--text_loss_strategy mse \
--text_loss_ratio 0.5 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=7;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language it \
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
--text_loss_strategy mse \
--text_loss_ratio 0.5 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=7;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language it \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.5 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=7;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language it \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 1e-4 \
--seed 1 \
--strategy baseline2 \
--text_loss_strategy mse \
--text_loss_ratio 0.5 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait
