.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = mp_identification
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


environment:
	conda env create -f environment.lock.yaml --force
	conda activate mp_perception2

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

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
