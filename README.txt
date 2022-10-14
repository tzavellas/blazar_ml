* code_clean -> fortran binaries
* code_clean.zip -> fortran binaries zipped
* hea_ml_generated_files/datasets -> ML datasets folder
* hea_ml_generated_files/inputs -> ML inputs that are used for generating datasets contents
* hea_ml_generated_files/saved_models -> Trained models
* hea_ml_generated_files/prediction -> predictions from the trained models for a 100 spectra (dataset_100)
* hea_ml.git -> Current local repo

* GAMMApy -> UNKNOWN


Fix docker permissions
----------------------
newgrp docker

Build docker
------------
docker build -f ./docker/Dockerfile ./docker -t hea_ml

Run docker
----------
docker run -v vol:/app/save_j1 hea_ml

Show docker volumes
-------------------
docker volume ls
docker volume inspect <vol>

Count csv columns
-----------------
head -1 file.csv | sed 's/[^,]//g' | wc -c



Prepare Inputs
--------------
python prepare_inputs.py -s 10000 -o ~/PROJECTS_MP/inputs/out10k.csv

Generate Dataset
----------------
python generate_dataset.py -e ~/PROJECTS_MP/code_clean/out -i ~/PROJECTS_MP/inputs/out10k.csv -w ~/PROJECTS_MP/datasets/dataset_10k

Interpolate
-----------
python interpolate_spectra.py -w ~/PROJECTS_MP/datasets/dataset_10k

Train
-----
python migrated/LSTM_network/training/mymodel_training.py ~/PROJECTS_MP/datasets/dataset_10k/normalized.csv saved_1k_b50 50

Predict
-------
python migrated/LSTM_network/prediction/mymodel_classification.py ~/PROJECTS_MP/inputs/out100.csv ~/PROJECTS_MP/hea_ml.git/saved_model ~/PROJECTS_MP/hea_ml.git/prediction/denormalized_100.csv 50

Evaluate
--------
python evaluation.py ~/PROJECTS_MP/datasets/dataset_100/interpolated.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b100/denormalized_100.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b100/rms_100.csv
