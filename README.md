# HEA ML Project

[comment]: <> (Foobar is a Python library for dealing with word pluralization.)

## Installation

### Docker
@todo

### Linux
Using conda/mamba:
```
mamba env create -f environment.yml
conda activate tf3
```

Using pip: @todo
```bash
# Install virtualenv
sudo apt install virtualenv

# Create a new environment
virtualenv hea_ml

# Activate environment
source hea_ml/bin/activate

# Install dependencies
pip install autopep8 astropy matplotlib numpy pandas scipy
```

## Usage
### Sample input parameter space
Create a CSV containing vectors of inputs.
```bash
python dataset/prepare_inputs.py -s <count> -o <path_to_csv>
```

### Generate Dataset
Create a `config.json` using a template under `config_files/dataset/template.json`. 
Update the file to use the `<path_to_csv>` of the previous step.
Then run:
```bash
python dataset/generate_dataset.py -c config.json
```

### Tuning
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

## Removal
### Linux
```bash
conda deactivate
rm -rf hea_ml
```
#

## Contributing

## License
[comment]: <> (MIThttps://choosealicense.com/licenses/mit/)
