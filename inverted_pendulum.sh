eval "$(conda shell.bash hook)"
conda activate gym_full

for i in 10000 30000 50000
do
	for j in 0.01 0.03 0.05
    do
        python main.py -env InvertedPendulum-v2 -s $i --discount 0.9 -n 100 -l 2 -s $i -b $i -lr $j --exp_name "IVP_b${i}_r${j}" -dev cpu &
    done
done

notify-send "inverted pendulum done"