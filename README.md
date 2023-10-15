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

Using pip:
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
Creates a CSV containing vectors of inputs.
```bash
python dataset/prepare_inputs.py -s count -o path_to_csv
```

### Generate Dataset
Create a `config` using a template under `config_files/dataset/template.json`. 
Update the file to use the `path_to_csv` of the previous step.
Then run:
```bash
python dataset/generate_dataset.py -c config
```

### Tuning



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
