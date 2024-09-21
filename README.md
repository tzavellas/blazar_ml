# HEA ML Project

[comment]: <> (Foobar is a Python library for dealing with word pluralization.)

## Hardware and Software Requirements
### Software
* TensorFlow: The source code is built using [Tensorflow](https://www.tensorflow.org/) 2.11, but it should be compatible with any TensorFlow version >= 2.11.
* Python: The code supports Python versions 3.9 to 3.12.
### GPU Support (Optional)
To enable GPU acceleration, the following NVIDIA® software is required:

* NVIDIA® GPU drivers:
    Version >= 525.60.13 for Linux
    Version >= 528.33 for WSL on Windows
* CUDA® Toolkit: Version 12.3
* cuDNN SDK: Version 8.9.7

To check your installed CUDA version and GPU drivers:
```
nvidia-smi
```

To check your current cuDNN SDK version:
```
nvcc --version
```

## Directory Structure
The project is organized into the following directories:
```
    .
    ├── config_files            # Configuration templates and settings
    ├── datasets                # Generated datasets
    ├── dataset                 # Scripts used for generating a dataset
    ├── ml                      # Machine learning model training and evaluation scripts
    ├── environment.yml         # Environment file
    ├── Dokerfile
    └── README.md
```

## Installation
### Linux
Using conda:
```
conda env create -f environment.yml
conda activate blazarml
```

## Usage
### Creating a sample from the input parameter space
The script `prepare_inputs.py` generates a sample of the input space. Below is the basic usage and description of its options.

```bash
python prepare_inputs.py [-h] [-s SIZE] [-o OUT]
```
#### Options
* -h, --help: Displays the help message and exits.
* -s SIZE, --size SIZE: Specifies the size of the sample (default is 10).
* -o OUT, --out OUT: Specifies the output file in CSV format (default is out.csv).

#### Example
1. Generate a sample with the default size and output file:
  ```
  python prepare_inputs.py
  ```
2. Generate a sample of 1000 entries and save it to a custom output file (e.g., sample.csv):
  ```
  python prepare_inputs.py -s 1000 -o sample.csv
  ```

### Generating a Dataset
1. Create a `config.json` file by copying the template found at `config_files/dataset/template.json`.
2. Update the `config.json` file to include the correct `OUT` csv from the previous step.
3. Once the `config.json` is updated, run the following command to generate the dataset:
```bash
python dataset/generate_dataset.py -c config.json
```

### Tuning a NN model
There are three types of neural network (NN) models that can be trained, depending on the type of NN layer used:

- **dnn**: A [typical densely-connected NN layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
- **gru**: A [Gated Recurrent Unit (GRU)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
- **lstm**: A [Long Short-Term Memory (LSTM) layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)

### Hyperparameter Tuning

Each model comes with a set of hyperparameters that can be adjusted. These hyperparameters significantly impact the convergence and performance of the model during training, so they must be carefully chosen. To optimize these hyperparameters, we use [Keras Tuner](https://keras.io/keras_tuner/), which automates the process of finding the best configuration for the model.


* Single run: Use a `config` file under `config_files/spectrum/tune`.
  ```bash
  python ml/tune.py -c <config>
  ```
* Single type run: Choose a `type` among the following `{dnn, rnn, gru, lstm}`. This will run all config files under `config_files/spectrum/tune/<type>`.
  ```bash
  ./train.sh -m tune -t <type>
  ```


### Training
* Single run: Use a `config` file under `config_files/spectrum/train`.
  ```bash
  python ml/train.py -c `<config>`
  ```
* Single type run: Choose a `type` among the following `{dnn, rnn, gru, lstm}`. This will run all config files under `config_files/spectrum/train/<type>`.
  ```bash
  ./train.sh -m train -t <type>
  ```

[comment]: <> ## License
[comment]: <> (MIThttps://choosealicense.com/licenses/mit/)
