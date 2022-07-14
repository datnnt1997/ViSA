# ViSA
Vietnamese sentiment analysis 


## RUN
```bash
python3 main.py train --task UIT-ViSD4SA --run_test --data_dir ./datasets/UIT-ViSD4SA --model_name_or_path vinai/phobert-base --model_arch crf --output_dir outputs --max_seq_length 128 --train_batch_size 32 --eval_batch_size 32 --learning_rate 1e-4 --classifier_learning_rate 3e-4 --epochs 100 --early_stop 5 --overwrite_data
```