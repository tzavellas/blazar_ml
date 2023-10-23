#!/bin/bash

# Function to print the help message
print_help() {
  echo "Usage: $0 -m <mode> -t <type> [-a <algorith>][-c <path>]"
  echo ""
  echo "Options:"
  echo "  -m    specify the mode (tune, train)"
  echo "  -t    specify the NN type (dnn, rnn, gru, lstm)"
  echo "  -a    specify tune algorithm (bayensian, grid, random)"
  echo "  -c    specify configuration path (default ../config_files/spectrum)"
  echo "  -h    display this help message and exit"
}

# Function to check if the current Python version is greater than the specified version
check_python_version() {
    local given_version=$1
    local current_version
    current_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    echo "Current Python version: $current_version"

    if [ "$(printf '%s\n' "$given_version" "$current_version" | sort -V | head -n1)" = "$given_version" ]; then
        if [ "$given_version" = "$current_version" ]; then
            echo "Python version is equal to $given_version. Exiting."
            exit 1
        else
            echo "Python version is greater than $given_version."
        fi
    else
        echo "Python version is less than $given_version. Exiting."
        exit 1
    fi
}

# Function to run a Python script with specific arguments
run_python_script() {
    local script_path=$1
    local config_file=$2
    
    if [ ! -f "$script_path" ]; then
        echo "Python script not found: $script_path"
        return 1
    fi
    echo "python $script_path -c $config_file"
    python "$script_path" -c "$config_file" >> output.txt 2>&1
}

CONFIG_PATH="../config_files/spectrum"

# Parse the options and arguments
while getopts "ht:m:a:c:" opt; do
  case $opt in
    h) # Display help
      print_help
      exit 0
      ;;
   m) # Set the mode if it is one of the expected values
      MODE=$OPTARG
      case $MODE in
        "train"|"tune")
          ;;
        *)
          echo "Unknown mode: $MODE"
          exit 1
          ;;
      esac
      ;;
    t) # Set the type if it is one of the expected values
      TYPE=$OPTARG
      case $TYPE in
        "dnn"|"rnn"|"gru"|"lstm")
          ;;
        *)
          echo "Unknown type: $TYPE"
          exit 1
          ;;
      esac
      ;;
    a) # Set the tune algorithm
      ALGORITHM=$OPTARG
      case $ALGORITHM in
        "bayensian"|"grid"|"random")
          ;;
        *)
          echo "Unknown algorithm"
          exit 1
          ;;
      esac
      ;;
    c) # Set the configuration path
      CONFIG_PATH=$OPTARG
      echo "Config Path: $CONFIG_PATH"
      ;;
    \?) # Invalid option
      echo "Invalid option: -$OPTARG" >&2
      print_help
      exit 1
      ;;
    :) # Missing argument
      echo "Option -$OPTARG requires an argument." >&2
      print_help
      exit 1
      ;;
  esac
done


check_python_version "3.7"

echo -e "\nMode: $MODE"
echo "NN Type: $TYPE"
MODE_PATH=$(readlink -f "$CONFIG_PATH"/"$MODE")
echo -e "using configuration path: $MODE_PATH\n"

# clean output.txt
echo -n > output.txt

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z "$ALGORITHM" ]; then
    for alg in "bayensian" "grid" "random"; do
        FILE=$(readlink -f "$MODE_PATH"/"$TYPE"/"$alg".json)
        if [ -f "$FILE" ]; then
            if ! run_python_script "$DIR/$MODE.py" "$FILE"; then
                echo -e "\nFile $FILE did not run."
                exit 1
            fi
        fi
    done
else
    FILE=$(readlink -f "$MODE_PATH"/"$TYPE"/"$ALGORITHM".json)
    if [ -f $FILE ]; then
        if ! run_python_script "$DIR/$MODE.py" "$FILE"; then
            echo -e "\nFile $FILE did not run."
            exit 1
        fi
    fi
fi
