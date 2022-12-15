# Automatic Differentiation Package
### Team 31
-------

[![.github/workflows/test.yml](https://code.harvard.edu/CS107/team31/actions/workflows/test.yml/badge.svg)](https://code.harvard.edu/CS107/team31/actions/workflows/test.yml)
[![.github/workflows/coverage.yml](https://code.harvard.edu/CS107/team31/actions/workflows/coverage.yml/badge.svg)](https://code.harvard.edu/CS107/team31/actions/workflows/coverage.yml)

### Contributors
-------
Cyrus Asgari, Caleb Saul, Sal Blanco, James Bardin

### Documentation
-------
[https://code.harvard.edu/CS107/team31/blob/main/docs/milestone2.ipynb](https://code.harvard.edu/CS107/team31/blob/main/docs/milestone2.ipynb)

### How to Install
-------
1. Navigate to desired directory and create virtual environment
```python
python -m venv test_env
```
2. Activate the environment 
```python
source test_env/bin/activate
```
3. Navigate inside test_env and install dependencies
```python
cd test_env
python -m pip install numpy
```
4. Install our package
```python
python -m pip install -i https://test.pypi.org/simple/ autodiff-team31==0.0.1
```
5. Write your code and import our package!
```python
>>>import autodiff as ad
>>>ad.sin(1)
0.8414709848078965
# More examples in documentation
```
6. Deactivate the environment
```python
deactivate
``` 