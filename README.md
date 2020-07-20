Predicting Perceived Naturalness of Human Animations Based on Generative Movement Primitive Models.
==============================

Data and analysis for Knopp, Benjamin, Dmytro Velychko, Johannes Dreibrodt, and Dominik Endres. 2019. “Predicting Perceived Naturalness of Human Animations Based on Generative Movement Primitive Models.” ACM Trans. Appl. Percept. 16 (3): 15:1–15:18. https://doi.org/10.1145/3355401.


The raw data is stored in directory `data/raw`. Please replicate the computing environment
using conda in an Unix-like operating system. Use the Makefile to process the raw data
through intermediate results into plots shown in the publication. The preprocessing is
slightly different, therefore the results will differ in an insignificantly.

1. Make sure to replicate the python environment using conda:

    `conda env create -f environment.lock.yaml --force`
    `conda activate mp_perception2`
    `pip install -e .`

2. Run `make all`. This will take a while (~15min on a fast computer). You can also 
    run processing scripts separately. Please refer to the `Makefile` to check dependencies
    of the processing, training, and visualization scripts.

If you have trouble executing these steps, please contact me: benjamin.knopp@uni-marburg.de.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
