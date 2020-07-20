#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = mp_identification
PYTHON_INTERPRETER = python3

all: fig2 fig3 fig4 fig5 fig6 fig7 fig8

regression: models/logistic_regression_model.pkl

models/logistic_regression_model.pkl:\
	src/models/train_model.py\
	data/processed/preprocessed_data.json
	$(PYTHON_INTERPRETER) src/models/train_model.py

fig2: reports/figures/fig2.pdf

fig3: reports/figures/fig3.pdf

fig4: reports/figures/fig4.pdf

fig5: reports/figures/fig5.pdf

fig6: reports/figures/fig6.pdf

fig7: reports/figures/fig7.pdf

fig8: reports/figures/fig8.pdf

reports/figures/fig8.pdf: data/processed/preprocessed_data.json
	$(PYTHON_INTERPRETER) src/visualization/make_fig8.py

reports/figures/fig7.pdf: data/processed/preprocessed_data.json\
	data/processed/joint_scores.json\
	models/logistic_regression_model.pkl
	$(PYTHON_INTERPRETER) src/visualization/make_fig7.py

reports/figures/fig6.pdf: data/processed/preprocessed_data.json\
	data/processed/joint_scores.json\
	models/logistic_regression_model.pkl
	$(PYTHON_INTERPRETER) src/visualization/make_fig6.py

reports/figures/fig5.pdf: data/processed/preprocessed_data.json\
	data/processed/joint_scores.json\
	models/logistic_regression_model.pkl
	$(PYTHON_INTERPRETER) src/visualization/make_fig5.py

reports/figures/fig3.pdf: data/processed/preprocessed_data.json\
	src/visualization/make_fig3and4.py
	$(PYTHON_INTERPRETER) src/visualization/make_fig3and4.py

reports/figures/fig4.pdf: data/processed/preprocessed_data.json\
	src/visualization/make_fig3and4.py
	$(PYTHON_INTERPRETER) src/visualization/make_fig3and4.py

reports/figures/fig2.pdf: data/processed/preprocessed_data.json\
	src/visualization/make_fig2.py
	$(PYTHON_INTERPRETER) src/visualization/make_fig2.py

## Make Dataset
data: data/processed/preprocessed_data.json data/processed/joint_scores.json

data/processed/joint_scores.json: data/processed/joint_results.json
	$(PYTHON_INTERPRETER) src/data/process_data.py

data/processed/preprocessed_data.json: data/processed/joint_results.json\
	src/data/process_data.py
	$(PYTHON_INTERPRETER) src/data/process_data.py

join: data/processed/joint_results.json

data/processed/joint_results.json:\
	data/interim/temp_exp1.json\
	data/interim/temp_exp2.json\
	data/interim/scores_exp1.json\
	data/interim/scores_exp2.json\
	src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py join data/interim data/processed

data/interim/temp_exp1.json: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py exp1 data/raw data/interim

data/interim/temp_exp2.json: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py exp2 data/raw data/interim

data/interim/scores_exp1.json: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py scores1 data/raw data/interim

data/interim/scores_exp2.json: src/data/make_dataset.py
	$(PYTHON_INTERPRETER) src/data/make_dataset.py scores2 data/raw data/interim

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
