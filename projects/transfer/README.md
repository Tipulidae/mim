# Transfer learning project
Welcome to the transfer learning project. Here you'll find all the code that was used for the article [Transfer Learning for Predicting Acute Myocardial Infarction Using Electrocardiograms](https://doi.org/).

As you may have noticed, this code repository is used for other projects as well, and some code is shared between projects. Please refer to the readme in the root folder for general instructions on how to use this repository. 

## Model names
Internally, we have used different names for the models than what was used in the final paper. 

| Paper | Code |
| --- | ---------|
| CNN-20k | CNN1 |
| RN-900k | XRN50A |
| RN-7M   | RN1 |
| RN-33M  | RN2 |

We have tested many things that were not ultimately included in the paper. All experiments are listed in experiments.py, and each experiment follows a certain naming convention. For the Target experiments: ```{source}_{model}_{target}```, where ```source``` indicates source outcome and size of the source training data, while ```target``` indicates the size of the target training dataset. Example:
```PTAS040_XRN50A_R070``` refers to the ```RN-900k``` model being first pre-trained on 40% of the source data to predict age and sex (AS), and then fine-tuned on 70% of the target data (R here refers to the raw ECG signal, as opposed to the median beat, which ultimately was never used in this project). 

For Source experiments, the naming convention was ```{model}_{data}_{outcome}```. For instance, ```RN1_R080_AGE_SEX``` refers to the ```RN-7M``` model being trained to predict age and sex using 80% of the source training data. 
