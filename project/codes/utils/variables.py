# This file contains common variables that are used throughout the project.
# libraries
import os

# paths and directories
run_local = True
if run_local:
    dataset_path = "/Users/masoud/Documents/Education/Alphanumerics Lab/Projects/data/physionet.org/files/ptbdb/1.0.0"
else:
    dataset_path = "/labs/samenilab/team/masoud_nateghi/data/physionet.org/files/ptbdb/1.0.0/"

os.makedirs("misc/models", exist_ok=True)
os.makedirs("misc/results", exist_ok=True)
os.makedirs("misc/dataset", exist_ok=True)

# variables
fs_old = 1000  # Sampling frequency
fc = 0.5  # Cut-off frequency
fs = 360  # New sampling frequency
