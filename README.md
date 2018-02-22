### Clone the Universial Dependencies treebanks:
    $ git clone https://github.com/UniversalDependencies/UD_Swedish-Talbanken.git

    $ git clone https://github.com/UniversalDependencies/UD_English-EWT.git

### Clone the baseline system:
    $ git clone git@gitlab.ida.liu.se:nlp2018-group3/baseline-system.git

    $ cd baseline-system

    $ python evaluate_on_english_treebank.py
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
    Parsing sentence #2001
    Tagging accuracy: 93.34%
    Unlabelled attachment score: 52.66%


    $ python evaluate_on_swedish_treebank.py
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
    Parsing sentence #503
    Tagging accuracy: 93.62%
    Unlabelled attachment score: 49.57%
