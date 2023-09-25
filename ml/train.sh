#python train.py -c ../config_files/spectrum/lstm_random.json
#python train.py -c ../config_files/spectrum/lstm_bayensian.json
#python train.py -c ../config_files/spectrum/lstm_grid.json

python train.py -c ../config_files/spectrum/gru_random.json
python train.py -c ../config_files/spectrum/gru_bayensian.json
python train.py -c ../config_files/spectrum/gru_grid.json

#python train.py -c ../config_files/spectrum/rnn_random.json
#python train.py -c ../config_files/spectrum/rnn_bayensian.json
#python train.py -c ../config_files/spectrum/rnn_grid.json

#python train.py -c ../config_files/spectrum/dnn_random.json
#python train.py -c ../config_files/spectrum/dnn_bayensian.json
#python train.py -c ../config_files/spectrum/dnn_grid.json
