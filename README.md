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
conda install -y autopep8 astropy matplotlib numpy pandas
```

### Linux
Using pip:
```bash
# Install virtualenv
sudo apt install virtualenv

# Create a new environment
virtualenv hea_ml

# Activate environment
source hea_ml/bin/activate

# Install dependencies
pip install autopep8 astropy matplotlib numpy pandas
```

## Usage
```bash
python src/prepare_inputs.py
python src/generate_dataset.py --executable /path/to/program --input out.csv
python src/plot_spectra.py --working-dir /path/to/dir --output spectra.png
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
