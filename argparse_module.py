"""
Module containing the argument parser for the script.

The purpose of this module is to provide a convenient and centralized way to
handle the command-line arguments required by the script. It uses the argparse
library to define and parse the following arguments:

MODEL: A required positional argument, which specifies the name of the model
to download from Hugging Face.
--branch: An optional argument, which specifies the name of the Git branch to
download from. The default value is 'main'.
--threads: An optional argument, which specifies the number of files to
download simultaneously. The default value is 1.

Example usage:
    python download-model.py MODEL --branch feature_branch --threads 4
"""


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('MODEL', type=str)
parser.add_argument('--branch', type=str, default='main', help='Name of the Git branch to download from.')
parser.add_argument('--threads', type=int, default=1, help='Number of files to download simultaneously.')
args = parser.parse_args()
