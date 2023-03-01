for seed in {1..5..1}
do
    python main.py --dataset SplitMNIST --method EWC --seed $seed
done