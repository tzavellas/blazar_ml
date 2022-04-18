# HEA ML Project

[comment]: <> (Foobar is a Python library for dealing with word pluralization.)

## Installation

### Windows
Using Anaconda:
```bash
# Create a new environment
conda create -y --name hea_ml

# Activate environment
conda activate hea_ml

# Install dependencies
conda install -y astropy numpy pandas
```

### Linux
Using pip:
```bash
# Install virtualenv
pip install virtualenv

# Create a new environment
virtualenv hea_ml

# Activate environment
source hea_ml/bin/activate

# Install dependencies
pip install -y astropy numpy pandas
```

## Usage
```bash
python src/generate_data.py
```

## Removal
### Windows
```bash
conda deactivate
conda remove --name hea_ml --all
```
### Linux
```bash
deactivate
rm -rf hea_ml
```
#

## Contributing

## License
[comment]: <> (MIThttps://choosealicense.com/licenses/mit/)
