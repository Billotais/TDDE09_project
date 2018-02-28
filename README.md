# Get up and running
This is a guide to get started as quickly as possible with the advanced system,
for a more detailed explanation of each class, refer to the
documentation.

## Setup
In order be able to use our advanced system it is necessary to
retrieve data from the Universal Dependencies treebanks. 
Therefore, clone the treebanks you are interested in.
Our example includes the English, the Swedish and the French dataset. 
Further languages can be found online in the UniversalDependencies repository.
If you want to get a feeling for how the system works feel free to see the 
section “Run” below.

### Dependencies
In order to be able to run the model it is of course necessary
to have all necessary packages installed. All of the packages needed come
with Python 3.6.

### Clone the Universial Dependencies treebanks:
`$ git clone https://github.com/UniversalDependencies/UD_Swedish-Talbanken.git`

`$ git clone https://github.com/UniversalDependencies/UD_English-EWT.git`

`$ git clone https://github.com/UniversalDependencies/UD_French-GSD.git`

### Clone the baseline system:
`$ git clone git@gitlab.ida.liu.se:nlp2018-group3/baseline-system.git`

`$ cd baseline-system`

## NN
Prepare data <br>
`$ python3 prepare_data.py ../UD_English-EWT/en-ud-train.conllu ../UD_English-
EWT/en-ud-dev.conllu ./train_config.json`
<br>Train <br>
`$ python3 train.py ./train_config.json`

## Run
In order to run the advanced system, simply run <br>
`$ python3 evaluate_system.py --train ../UD_English-EWT/en-ud-train.conllu ../UD_English-EWT/en-ud-dev.conllu` 
<br>for the English dataset,<br>
`$ python3 evaluate_system.py --train ../UD_Swedish-Talbanken/sv-ud-train.conllu ../UD_Swedish-Talbanken/sv-ud-dev.conllu`
<br>for the Swedish dataset and<br>
`$ python3 evaluate_system.py --train ../UD_French-GSD/fr-ud-train.conllu ../UD_French-GSD/fr-ud-dev.conllu`
<br>for the French dataset. The function loads the training data, trains the tagger, 
trains the parser based on the trained tagger and finally evaluates the system on the test dataset. In the following the results for the English and the Swedish treebank can be found:<br>

### Save and load models
To train and save a model
`$ python evaluate_system.py --train ../UD_English-EWT/en-ud-train.conllu ../UD_English-EWT/en-ud-dev.conllu --save english.trained`
To load a trained mode
`$ python evaluate_system.py ../UD_English-EWT/en-ud-dev.conllu --load english.trained`

English Treebank:

    Training POS tagger
    Epoch: 1 / 3
    Updated with sentence #12542
    Epoch: 2 / 3
    Updated with sentence #12542
    Epoch: 3 / 3
    Updated with sentence #12542
    Training syntactic parser:
    Epoch: 1 / 3
    Updated with sentence #12542
    Epoch: 2 / 3
    Updated with sentence #12542
    Epoch: 3 / 3
    Updated with sentence #12542
    Evaluation:
    Tagging accuracy: 93.26%
    Unlabelled attachment score: 65.29%

Swedish Treebank:

    Training POS tagger
    Epoch: 1 / 3
    Updated with sentence #4302
    Epoch: 2 / 3
    Updated with sentence #4302
    Epoch: 3 / 3
    Updated with sentence #4302
    Training syntactic parser:
    Epoch: 1 / 3
    Updated with sentence #4302
    Epoch: 2 / 3
    Updated with sentence #4302
    Epoch: 3 / 3
    Updated with sentence #4302
    Evaluation:
    Tagging accuracy: 93.69%
    Unlabelled attachment score: 59.65%

French Treebank:

    Training POS tagger
    Epoch: 1 / 3
    Updated with sentence #14553
    Epoch: 2 / 3
    Updated with sentence #14553
    Epoch: 3 / 3
    Updated with sentence #14553
    Training syntactic parser:
    Epoch: 1 / 3
    Updated with sentence #14553
    Epoch: 2 / 3
    Updated with sentence #14553
    Epoch: 3 / 3
    Updated with sentence #14553
    Evaluation:
    Tagging accuracy: 96.57%
    Unlabelled attachment score: 81.33%



# Abstract

Feature engineering on the parser : 

- EN : 65.29% to 69.33%
- SV : 59.65% to 64.32%

Use of a non-projectivize parser :

- EN : 69.33% to 69.10%
- SV : 64.32% to 64.93%

Further improvement of the features : 

- EN : 69.10% to 69.38%
- SV : 64.93% to 65.14%

Replaced the greedy search by a beam search : 

- No improvement in accuracy

Use features like left-most child, ...

- EN : 69.38% to 76.70%
- SV : 65.14% to 71.79%

Few tweaks in features

- EN : 76.70% to 76.24%
- SV : 71.79% to 72.45%

Added French treebank for more tests

- FR : 81.33% accuracy
