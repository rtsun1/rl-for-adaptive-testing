# conda env update -f environment.yml --prune
# conda activate cat

time_limit_s=$((60*15))  # 15 minutes
eval_size=1000  # 1000 students to evaluate
seed=42

train_steps=500000  # steps to train dqn

default_args="--time-limit-s $time_limit_s --eval-episodes $eval_size --seed $seed"
python dqn.py --total-timesteps $train_steps --detailed-answered --exp-name "dqn_danic_rep_${train_steps}_train_steps_${eval_size}_students" $default_args
# # Greedy
# echo "Running Greedy"
# python greedy.py --no-time-aware --exp-name "greedy_${eval_size}_students" $default_args
# python greedy.py --time-aware --exp-name "greedy_time_aware_${eval_size}_students" $default_args

# # DQN
# echo "Running DQN"
# python dqn.py --total-timesteps $train_steps --exp-name "dqn_${train_steps}_train_steps_${eval_size}_students" $default_args
# python dqn.py --total-timesteps $train_steps --mask-answered --exp-name "dqn_mask_${train_steps}_train_steps_${eval_size}_students" $default_args
# python dqn.py --total-timesteps $train_steps --include-answered --exp-name "dqn_include_answered_${train_steps}_train_steps_${eval_size}_students" $default_args