# cpu_num=(  )
(
i=2;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language en \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=4;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language de \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
# cpu_idx=cpu_idx+1
(
i=5;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language fr \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=6;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language es \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
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
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=7;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language ko \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait
sleep 10s
(
i=4;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language pl \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=5;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language ru \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=6;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language tr \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
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
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
(
i=7;
export CUDA_VISIBLE_DEVICES=${i}
python align_from_to_en/run_cross.py \
--language jp \
--delta_tuning \
--lora_r 1 \
--n_shot 50 \
--epoch 50 \
--train_bs 5 \
--eval_bs 100 \
--eval_steps 5 \
--text_lr 3e-4 \
--seed 1 \
--strategy baseline1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait