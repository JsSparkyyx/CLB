for seed in {1..5..1}
do
    python main.py --dataset PMNIST --method EWC --seed $seed --train_batch_size 2048 --test_batch_size 2048
done