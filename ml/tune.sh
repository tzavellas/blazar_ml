#python tune.py -c ../config_files/spectrum/tune_lstm_random.json
#python tune.py -c ../config_files/spectrum/tune_lstm_bayensian.json
#python tune.py -c ../config_files/spectrum/tune_lstm_grid.json

python tune.py -c ../config_files/spectrum/tune_gru_random.json
python tune.py -c ../config_files/spectrum/tune_gru_bayensian.json
python tune.py -c ../config_files/spectrum/tune_gru_grid.json

#python tune.py -c ../config_files/spectrum/tune_rnn_random.json
#python tune.py -c ../config_files/spectrum/tune_rnn_bayensian.json
#python tune.py -c ../config_files/spectrum/tune_rnn_grid.json

#python tune.py -c ../config_files/spectrum/tune_dnn_random.json
#python tune.py -c ../config_files/spectrum/tune_dnn_bayensian.json
#python tune.py -c ../config_files/spectrum/tune_dnn_grid.json
