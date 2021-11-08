eval "$(conda shell.bash hook)"
conda activate gym_full

python main.py -env LunarLanderContinuous-v2 -el 1000 --discount 0.99 -n 100 -l 2 -nh 64 -s 40000 -b 20000 -lr 0.005 --exp_name ll_s40000_b20000_r.005 -dev cpu