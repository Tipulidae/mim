#!/bin/bash
echo "Creating or updating conda environments. There are separate environments
depending on if you want to use tensorflow with or without gpu. Select which
environments to create: (1) CPU only (2) GPU (3) Both CPU and GPU environments"

read answer

if [[ $answer = 1 ]]
then
  conda env update --name mim-cpu --file environment_cpu.yml --prune
elif [[ $answer = 2 ]]
then
  conda env update --name mim --file environment.yml --prune
else
  conda env update --name mim-cpu --file environment_cpu.yml --prune
  conda env update --name mim --file environment.yml --prune
fi
