# conda env update -f environment.yml --prune
# conda activate cat

time_limit_s=$((60*15))  # 15 minutes
eval_size=100  # 100 students to evaluate
seed=42
num_envs=4

train_steps=125000  # steps to train dqn

default_args="--time-limit-s $time_limit_s --eval-episodes $eval_size --seed $seed"

# Greedy
echo "Running Greedy"
python greedy.py --no-time-aware --exp-name "greedy_${eval_size}_students" $default_args
python greedy.py --time-aware --exp-name "greedy_time_aware_${eval_size}_students" $default_args

# DQN
echo "Running DQN"
dqn_args="$default_args --num-envs $num_envs --total-timesteps $train_steps --no-cuda"  # on mac, with small Q net, cpu is faster
python dqn.py --exp-name "dqn_${train_steps}_train_steps_${eval_size}_students" $dqn_args
python dqn.py --mask-answered --exp-name "dqn_mask_${train_steps}_train_steps_${eval_size}_students" $dqn_args
python dqn.py --include-answered --exp-name "dqn_include_answered_${train_steps}_train_steps_${eval_size}_students" $dqn_args
python dqn.py --detailed-answered --exp-name "dqn_detailed_answered_${train_steps}_train_steps_${eval_size}_students" $dqn_args
