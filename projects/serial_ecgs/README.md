# Serial ECG project
Welcome to the serial ECG project. Here you'll find all the code that was 
used for the article *Prior electrocardiograms not useful for predicting 
major adverse cardiac events using machine learning*.  

As you may have noticed, this code repository is used for other projects as 
well, and some code is shared between projects. 

Please refer to the readme in the root folder for general instructions on 
how to use this repository. 

## The final models
The final, trained models are too large to store on github. Instead, you 
can download them from 
<a href="https://lu.box.com/s/2u2q3txlzbed8tilllw8zcorghmulgz1">here</a>.

The archive contains the 12 neural network models (MLP, CNN and RN 
models for each of the 4 input sets). Each model type is numbered 1-4 
depending on its expected input:

| Number | Inputs |
| --- | ---------|
| 1 | Index ECG |
| 2 | Index ECG + prior ECG |
| 3 | Index ECG + additional clinical variables |
| 4 | Index ECG + prior ECG + addititional clinical variables |

For each model and input, there's a folder containing 10 models of 
identical structure, but where the training started with a different weight 
initialization, as described in the article. If, for instance, you wanted to 
load the first CNN model trained with index ECG and additional clinical 
variables (input set number 3), it would look like this:

<pre><code>from tensorflow import keras
cnn3 = keras.models.load_model(filepath='final_models/CNN3/1')
</code></pre>

In the article, the final output was taken as the average of the output 
of all ten models, for each model type and input set. 

## Latex
The latex code for generating the article and most of the figures is in the 
latex folder. 