#!/bin/bash
echo "Updating your conda environment..."

conda env update --name mim --file environment.yml --prune

echo "Done! Activate the environment by typing 'conda activate mim'."
