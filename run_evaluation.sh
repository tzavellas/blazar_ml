#!/bin/sh
cd dataset

python evaluation.py ~/PROJECTS_MP/datasets/dataset_100/interpolated.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b50/denormalized_100.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b50/rms_100.csv

python evaluation.py ~/PROJECTS_MP/datasets/dataset_100/interpolated.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b75/denormalized_100.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b75/rms_100.csv

python evaluation.py ~/PROJECTS_MP/datasets/dataset_100/interpolated.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b100/denormalized_100.csv ~/PROJECTS_MP/hea_ml.git/prediction_1k_b100/rms_100.cs

python evaluation.py ~/PROJECTS_MP/datasets/dataset_100/interpolated.csv ~/PROJECTS_MP/hea_ml.git/prediction_10k_b50/denormalized_100.csv ~/PROJECTS_MP/hea_ml.git/prediction_10k_b50/rms_100.csv

python evaluation.py ~/PROJECTS_MP/datasets/dataset_100/interpolated.csv ~/PROJECTS_MP/hea_ml.git/prediction_10k_b100/denormalized_100.csv ~/PROJECTS_MP/hea_ml.git/prediction_10k_b100/rms_100.csv
