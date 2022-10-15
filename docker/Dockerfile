FROM continuumio/miniconda3:latest

WORKDIR /app

# Create the environment
COPY ./environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "tf", "/bin/bash", "-c"]

# verify tensorflow installation
RUN python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# The code to run when container is started:
COPY pattern_prediction_NN_Shangying ./pattern_prediction

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf", "python", "pattern_prediction/LSTM_network/training/mymodel_training.py", "pattern_prediction/sample_data/all_data.csv"]
