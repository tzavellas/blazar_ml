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
docker build -f Dockerfile . --tag tf2 --output ../../docker-tf2

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
python dataset/prepare_inputs.py -s 10000 -o ~/PROJECTS_MP/inputs/out10k.csv

Generate Dataset
----------------
python dataset/generate_dataset.py -c config_file

Interpolate
-----------
python dataset/interpolate_spectra.py -c config_file

Train
-----
python ml/tune.py -c config_file
python ml/train.py -c config_file
OR
sh ml/train.sh -m <mode> -t <type> [-c <config_path>]

