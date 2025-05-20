.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# CLEAN TARGETS

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -f docs/api/*.rst
	-$(MAKE) -C docs clean 2>/dev/null  # this fails if sphinx is not yet installed

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage clean-docs ## remove all build, test, coverage, docs and Python artifacts


# INSTALL TARGETS

.PHONY: install
install: clean ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-dev
install-dev: clean ## install the package in editable mode with development dependencies
	pip install -e .[dev]

.PHONY: install-test
install-test: clean ## install the package with test dependencies
	pip install -e .[test]

.PHONY: install-pretrained
install-pretrained: clean-build clean-pyc ## install the package and pretrained pipelines
	pip install -e .[pretrained]

.PHONY: install-pretrained-develop
install-pretrained-develop: clean-build clean-pyc ## install the package with pretrained in editable mode and dependencies for development
	pip install -e .[pretrained-dev]

MINIMUM := $(shell sed -n '/install_requires = \[/,/]/p' setup.py | grep -v -e '[][]' | sed 's/ *\(.*\),$?$$/\1/g' | tr '>' '=')

.PHONY: install-minimum
install-minimum: ## install the minimum supported versions of the package dependencies
	pip install $(MINIMUM)

.PHONY: check-dependencies
check-dependencies: ## test if there are any broken dependencies
	pip check

# LINT TARGETS

.PHONY: lint
lint: ## check style with flake8 and isort
	invoke lint

.PHONY: lint-cap
lint-cap: ## check style with flake8 and isort
	flake8 cap tests
	isort -c --recursive cap tests

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find cap -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive cap
	isort --apply --atomic --recursive cap

	find tests -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive tests
	isort --apply --atomic --recursive tests


# TEST TARGETS

.PHONY: test
test: ## run tests
	pytest tests/

.PHONY: test-coverage
test-coverage: ## run tests with coverage
	coverage run --source cap -m pytest
	coverage report -m

.PHONY: test-minimum
test-minimum: install-minimum check-dependencies test ## run tests using the minimum supported dependencies

.PHONY: test-all
test-all: ## run tests on every Python version with tox
	tox -r

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source cap -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html


# DOCS TARGETS

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs html

.PHONY: view-docs
view-docs: ## view the docs in a browser
	$(BROWSER) docs/_build/html/index.html

.PHONY: serve-docs
serve-docs: view-docs ## compile the docs watching for changes
	watchmedo shell-command -W -R -D -p '*.rst;*.md' -c '$(MAKE) -C docs html' docs


# RELEASE TARGETS

.PHONY: dist
dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

.PHONY: publish
publish: dist ## publish to PyPI
	twine upload dist/*

.PHONY: publish-confirm
publish-confirm:
	@echo "WARNING: This will irreversibly upload a new version to PyPI!"
	@echo -n "Please type 'confirm' to proceed: " \
		&& read answer \
		&& [ "$${answer}" = "confirm" ]

.PHONY: publish-test
publish-test: dist publish-confirm ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: bumpversion-release
bumpversion-release: ## Merge master to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff master -m"make release-tag: Merge branch 'master' into stable"
	bumpversion release
	git push --tags origin stable

.PHONY: bumpversion-release-test
bumpversion-release-test: ## Merge master to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff master -m"make release-tag: Merge branch 'master' into stable"
	bumpversion release --no-tag
	@echo git push --tags origin stable

.PHONY: bumpversion-patch
bumpversion-patch: ## Merge stable to master and bumpversion patch
	git checkout master
	git merge stable
	bumpversion --no-tag patch
	git push

.PHONY: bumpversion-candidate
bumpversion-candidate: ## Bump the version to the next candidate
	bumpversion candidate --no-tag

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bumpversion --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bumpversion --no-tag major

.PHONY: bumpversion-revert
bumpversion-revert: ## Undo a previous bumpversion-release
	git checkout master
	git branch -D stable

CLEAN_DIR := $(shell git status --short | grep -v ??)
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
CHANGELOG_LINES := $(shell git diff HEAD..origin/stable HISTORY.md 2>&1 | wc -l)

.PHONY: check-clean
check-clean: ## Check if the directory has uncommitted changes
ifneq ($(CLEAN_DIR),)
	$(error There are uncommitted changes)
endif

.PHONY: check-master
check-master: ## Check if we are in master branch
ifneq ($(CURRENT_BRANCH),master)
	$(error Please make the release from master branch\n)
endif

.PHONY: check-history
check-history: ## Check if HISTORY.md has been modified
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: check-release
check-release: check-master check-history ## Check if the release can be made
	@echo "A new release can be made"

.PHONY: release
release: check-release bumpversion-release publish bumpversion-patch

.PHONY: release-test
release-test: check-release bumpversion-release-test publish-test bumpversion-revert

.PHONY: release-candidate
release-candidate: check-master publish bumpversion-candidate

.PHONY: release-candidate-test
release-candidate-test: check-clean check-master publish-test

.PHONY: release-minor
release-minor: check-release bumpversion-minor release

.PHONY: release-major
release-major: check-release bumpversion-major release


# DOCKER TARGETS

.PHONY: docker-jupyter-clean
docker-jupyter-clean: ## Remove the cap-jupyter docker image
	docker rmi -f cap-jupyter

.PHONY: docker-jupyter-build
docker-jupyter-build: ## Build the cap-jupyter docker image using repo2docker
	docker build -f docker/cap-jupyter.Dockerfile -t cap-jupyter .

.PHONY: docker-jupyter-save
docker-jupyter-save: docker-jupyter-build  ## Build the cap-jupyter image and save it as cap-jupyter.tar
	docker save --output cap-jupyter.tar cap-jupyter

.PHONY: docker-jupyter-load
docker-jupyter-load: ## Load the cap-jupyter image from cap-jupyter.tar
	docker load --input cap-jupyter.tar

.PHONY: docker-jupyter-run
docker-jupyter-run: ## Run the cap-jupyter image in editable mode
	docker run --rm \
		-v $(shell pwd)/cap:/app/cap \
		-v $(shell pwd)/tutorials:/app/tutorials \
		-ti -p8888:8888 --name cap-jupyter cap-jupyter

.PHONY: docker-jupyter-start
docker-jupyter-start: ## Start the cap-jupyter image as a daemon
	docker run --rm -d -v $(shell pwd):/app -ti -p8888:8888 --name cap-jupyter cap-jupyter

.PHONY: docker-jupyter-stop
docker-jupyter-stop: ## Stop the cap-jupyter daemon
	docker stop cap-jupyter

.PHONY: docker-distribute
docker-jupyter-package: docker-jupyter-save ## Build the docker-jupyter image and package it for distribution
	docker/package.sh


# DEPENDENCY TARGETS

.PHONY: checkdeps
checkdeps: # Save the currently installed versions of the dependencies as the latest versions 
	invoke checkdeps $(OUTPUT_PATH)

# MODEL TARGETS

.PHONY: run-transformer
run-transformer: ## run transformer model example
	python -m cap.models.transformer

.PHONY: run-fedformer
run-fedformer: ## run FEDFormer model example
	python -m cap.models.FEDFormer

.PHONY: run-autoformer
run-autoformer: ## run Autoformer model example
	python -m cap.models.Autoformer

.PHONY: run-timesnet
run-timesnet: ## run TimesNet model example
	python -m cap.models.TimesNet

.PHONY: run-informer
run-informer: ## run Informer model example
	python -m cap.models.Informer

.PHONY: run-lstm
run-lstm: ## run LSTM model example
	python -m cap.models.lstm
