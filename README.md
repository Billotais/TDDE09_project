
# Abstract

From the baseline system, we decided to develop two different independent systems. 

On one side, we tried to improve the accuracy of the parser by changing some aspect of the training and prediction algorithms, but we kept the same simple percepton. We implemented better features and non-projective tree support with the SWAP operation We also tried to use beam-search instead of greedy-search during both training and parsing, but we got some unexpected results. We experimented with arc-eager transition based system instead of arc-standard and with sentence generation to provide the parser with a bigger training data set, but we decided not to include theses changes for various reasons.

In parallel, we developped ...

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

`$ git clone https://github.com/UniversalDependencies/UD_German-GSD.git`

### Clone the advances system:
`$ git clone git@gitlab.ida.liu.se:nlp2018-group3/advanced-system.git`

`$ cd advanced-system`

## Neural Network using Tensorflow
Prepare data <br>
`$ python3 prepare_data.py ../UD_English-EWT/en-ud-train.conllu ../UD_English-EWT/en-ud-dev.conllu ../UD_English-EWT/en-ud-test.conllu ./train_config.json`
<br>Train <br>
`$ python3 train.py ./train_config.json`

## Run the parser
In order to run the advanced system, simply run <br>
`$ python3 evaluate_system.py --train ../UD_English-EWT/en-ud-train.conllu ../UD_English-EWT/en-ud-dev.conllu` 
<br>for the English dataset,<br>
`$ python3 evaluate_system.py --train ../UD_Swedish-Talbanken/sv-ud-train.conllu ../UD_Swedish-Talbanken/sv-ud-dev.conllu`
<br>for the Swedish dataset,<br>
`$ python3 evaluate_system.py --train ../UD_French-GSD/fr-ud-train.conllu ../UD_French-GSD/fr-ud-dev.conllu`
<br>for the French dataset and<br>
`$ python evaluate_system.py --train ../UD_German-GSD/de-ud-train.conllu ../UD_German-GSD/de-ud-dev.conllu`
<br>for the German dataset.

The function loads the training data, trains the tagger, 
trains the parser based on the trained tagger and finally evaluates the system on the test dataset.

### Save and load models
To train and save a model
`$ python evaluate_system.py --train ../UD_English-EWT/en-ud-train.conllu ../UD_English-EWT/en-ud-dev.conllu --save english.trained`

To load a trained mode
`$ python evaluate_system.py ../UD_English-EWT/en-ud-dev.conllu --load english.trained`

### Available options
`--train file` To train the parser using training data

`--load file` To load a trained parser from a file

`--save file` To save the trained parser to a file

`--trunc_data n` To only train on the n first sentences

`--n_epochs n` Set the number of training epochs 

`--beam_size n` Set the size of the beam during evaluation



We got the following results with our parser using the simple Perceptron : 

#### English (on en-ud-dev.conllu)

| Beam size  |Tagging accuracy|Unlabelled attachement score|Exact Matches|
|------------|----------------|----------------------------|-------------|
|1           |93.34%          |76.11%                      |41.66%       |
|2           |93.34%          |76.11%                      |42.11%       |
|8           |93.34%          |75.40%                      |40.81%       |
|16          |93.34%          |75.37%                      |40.21%       |
|32          |93.34%          |74.52%                      |39.26%       |

#### Swedish

| Beam size  |Tagging accuracy|Unlabelled attachement score|Exact Matches|
|------------|----------------|----------------------------|-------------|
|1           |93.62%          |71.98%                      |17.78%       |
|2           |93.62%          |71.66%                      |18.25%       |
|8           |93.62%          |70.23%                      |17.66%       |
|16          |93.62%          |69.44%                      |16.87%       |
|32          |93.62%          |68.14%                      |16.27%       |

#### French
| Beam size  |Tagging accuracy|Unlabelled attachement score|Exact Matches|
|------------|----------------|----------------------------|-------------|
|1           |96.57%          |81.22%                      |16.71%       |
|2           |96.57%          |81.56%                      |17.12%       |
|4           |96.57%          |81.29%                      |17.25%       |
|8           |96.57%          |81.11%                      |16.64%       |
|16          |96.57%          |80.90%                      |16.91%       |










## (Old values with not correct beam search)

#### English
| Beam size  |Tagging accuracy|Unlabelled attachement score|Exact Matches|
|------------|----------------|----------------------------|-------------|
|1           |93.34%          |76.26%                      |41.41%       |
|2           |93.34%          |76.63%                      |42.01%       |
|8           |93.34%          |76.65%                      |42.16%       |
|32          |93.34%          |76.63%                      |42.16%       |
|64          |93.34%          |76.66%                      |42.16%       |

#### French (on fr-ud-dev.conllu)

| Beam size  |Tagging accuracy|Unlabelled attachement score|Exact Matches|
|------------|----------------|----------------------------|-------------|
|1           |96.57%          |81.34%                      |16.85%       |
|2           |96.57%          |81.78%                      |17.52%       |
|8           |96.57%          |81.90%                      |18.00%       |
|32          |96.57%          |81.92%                      |18.00%       |
|64          |96.57%          |81.93%                      |18.00%       |

#### Swedish (on sv-ud-dev.conllu)

| Beam size  |Tagging accuracy|Unlabelled attachement score|Exact Matches|
|------------|----------------|----------------------------|-------------|
|1           |93.62%          |72.43%                      |18.06%       |
|2           |96.57%          |72.53%                      |18.45%       |
|8           |96.57%          |72.62%                      |18.65%       |
|32          |96.57%          |72.45%                      |18.65%       |
|64          |96.57%          |72.59%                      |18.65%       |


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
