for seed in {1..5..1}
do
    python main.py --dataset CIFAR100 --method EWC --seed $seed --train_batch_size 256
done