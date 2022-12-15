# File       : test_operators.py
# Description: Test cases elementary operators
# Copyright 2022 Harvard University. All Rights Reserved.
import pytest
import numpy as np
import autodiff as ad
from autodiff.dualnumber import DualNumber


def test_sin():
    z1 = DualNumber(1,2)
    z2 = ad.sin(z1)
    assert z2.real == np.sin(1)
    assert z2.dual == 2*np.cos(1)
    assert ad.sin(4) == np.sin(4)
    assert ad.sin(0.5) == np.sin(0.5)
    with pytest.raises(TypeError):
        ad.sin("1")

def test_cos():
    z1 = DualNumber(1,2)
    z2 = ad.cos(z1)
    assert z2.real == np.cos(1)
    assert z2.dual == -2*np.sin(1)
    assert ad.cos(4) == np.cos(4)
    assert ad.cos(0.5) == np.cos(0.5)
    with pytest.raises(TypeError):
        ad.cos("1")

def test_tan():
    z1 = DualNumber(1,2)
    z2 = ad.tan(z1)
    assert z2.real == np.tan(1)
    assert z2.dual == 2/(np.cos(1)**2)
    assert ad.tan(4) == np.tan(4)
    assert ad.tan(0.5) == np.tan(0.5)
    with pytest.raises(TypeError):
        ad.tan("1")

def test_exp():
    z1 = DualNumber(1,2)
    z2 = ad.exp(z1)
    assert z2.real == np.exp(1)
    assert z2.dual == 2*np.exp(1)
    assert ad.exp(4) == np.exp(4)
    assert ad.exp(0.5) == np.exp(0.5)
    with pytest.raises(TypeError):
        ad.exp("1")

def test_log():
    z1 = DualNumber(1,2.)
    z2 = ad.log(z1)
    assert z2.real == np.log(1)
    assert z2.dual == 2.
    assert ad.log(4) == np.log(4)
    assert ad.log(0.5) == np.log(0.5)
    with pytest.raises(TypeError):
        ad.log("1")