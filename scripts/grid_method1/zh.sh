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
--text_lr 5e-5 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.3 \
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
--text_lr 1e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.3 \
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
--text_lr 2e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.3 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait
sleep 10s

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
--text_lr 5e-5 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
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
--text_lr 1e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
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
--text_lr 2e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.1 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait
sleep 10s

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
--text_lr 5e-5 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.03 \
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
--text_lr 1e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.03 \
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
--text_lr 2e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.03 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait
sleep 10s

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
--text_lr 5e-5 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.01 \
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
--text_lr 1e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.01 \
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
--text_lr 2e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.01 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait
sleep 10s

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
--text_lr 5e-5 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.003 \
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
--text_lr 1e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.003 \
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
--text_lr 2e-4 \
--seed 1 \
--strategy method1 \
--text_loss_strategy mse \
--text_loss_ratio 0.003 \
--output_basedir /data1/zhenzhang/dir2/align_from_to_en/outputs
)&
wait
sleep 10s