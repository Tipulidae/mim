#!/usr/bin/env bash

pdflatex model-lr1.tex
pdflatex model-lr2.tex
pdflatex model-lr3.tex
pdflatex model-lr4.tex
pdflatex model-mlp1.tex
pdflatex model-mlp2.tex
pdflatex model-mlp3.tex
pdflatex model-mlp4.tex
pdflatex model-cnn1.tex
pdflatex model-cnn2.tex
pdflatex model-cnn3.tex
pdflatex model-cnn4.tex
pdflatex model-rn1.tex
pdflatex model-rn2.tex
pdflatex model-rn3.tex
pdflatex model-rn4.tex
pdflatex overview.tex

pdfcrop model-lr1.pdf model-lr1.pdf
pdfcrop model-lr2.pdf model-lr2.pdf
pdfcrop model-lr3.pdf model-lr3.pdf
pdfcrop model-lr4.pdf model-lr4.pdf
pdfcrop model-mlp1.pdf model-mlp1.pdf
pdfcrop model-mlp2.pdf model-mlp2.pdf
pdfcrop model-mlp3.pdf model-mlp3.pdf
pdfcrop model-mlp4.pdf model-mlp4.pdf
pdfcrop model-cnn1.pdf model-cnn1.pdf
pdfcrop model-cnn2.pdf model-cnn2.pdf
pdfcrop model-cnn3.pdf model-cnn3.pdf
pdfcrop model-cnn4.pdf model-cnn4.pdf
pdfcrop model-rn1.pdf model-rn1.pdf
pdfcrop model-rn2.pdf model-rn2.pdf
pdfcrop model-rn3.pdf model-rn3.pdf
pdfcrop model-rn4.pdf model-rn4.pdf
pdfcrop overview.pdf overview.pdf


