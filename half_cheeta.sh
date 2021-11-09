eval "$(conda shell.bash hook)"
conda activate gym_full

python main.py -env HalfCheetah-v2 -el 150 --discount 0.95 -n 1000 -l 2 -nh 32 -s 50000 -b 25000 -lr 0.02 --exp_name hc_s50000_b25000_r0.02_n1000_2
