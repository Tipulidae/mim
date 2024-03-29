all: githooks setup conda

githooks: .hooks
	chmod +x .hooks/*
	CWD=`pwd`; for FILEPATH in .hooks/* ; do FILENAME=$${FILEPATH##*/}; ln -sf $$CWD/.hooks/$$FILENAME $$CWD/.git/hooks/$$FILENAME ; done
	git config --local include.path ../.gitconfig
	pre-commit install

setup: .scripts/setup.sh
	chmod +x .scripts/*
	./.scripts/setup.sh

pep:
	autopep8 -i -r mim/

conda: .scripts/conda.sh
	./.scripts/conda.sh
