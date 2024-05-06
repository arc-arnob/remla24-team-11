REMLA24-TEAM-11
==============================

## Local Development

### Prerequisites
You will need to have `poetry` installed on your machine. If you don't have it installed, you can install it by following the instructions [here](https://python-poetry.org/docs/). In addition to that, you will need Python 3.11 installed on your machine. You can install it by following the instructions here. To set poetry to use Python 3.11, run the following command `poetry env use 3.11`

### Setup
1. Clone the repository `git clone git@github.com:arc-arnob/remla24-team-11.git`
2. Run `poetry install --with dev` to install the dependencies
3. Run `poetry shell` to activate the virtual environment

### Lint
To lint the code, run the following command `pre-commit run --all-files`
