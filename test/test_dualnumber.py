# File       : test_dualnumber.py
# Description: Test cases for testing Dual numbers basic operations
# Copyright 2022 Harvard University. All Rights Reserved.
import pytest
import numpy as np
from autodiff.dualnumber import DualNumber


class TestDualNumber:
    """Test class for dual number types"""

    def test_init(self):
        z = DualNumber(2,3)
        assert isinstance(z, DualNumber)
        assert z.real == 2
        assert z.dual == 3
        z1 = DualNumber(1.3, 5.6)
        assert isinstance(z1, DualNumber)
        assert z1.real == 1.3
        assert z1.dual == 5.6
        z2 = DualNumber(3)
        assert z2.real == 3
        assert z2.dual == 1
        with pytest.raises(TypeError):
            DualNumber("5" , 2)
            DualNumber(2 , "2")

    def test_repr(self):
        z = DualNumber(2, 3)
        assert z.__repr__() == f"DualNumber(real: {2:e}, dual: {3:e})"

    def test_eq(self):
        z1 = DualNumber(2, 3)
        z2 = DualNumber(2, 3)
        assert z1 == z2
        z3 = DualNumber(2,4)
        assert z1 != z3
        with pytest.raises(TypeError):
            z1 == 4
    
    def test_neg(self):
        z1 = DualNumber(2,3)
        z2 = -z1
        assert z2.real == -2 and z2.dual == -3


    def test_addition(self):
        z1 = DualNumber(1,2)
        z2 = DualNumber(4,5)
        z3 = z1 + z2
        z4 = z1 + 1
        z5 = z2 + 4.5
        z6 = 1 + z1
        z7 = 4.5 + z2
        assert z3.real == 5
        assert z3.dual == 7
        assert z4.real == 2 
        assert z4.dual == 2
        assert z5.real == 8.5
        assert z5.dual == 5
        assert z6.real == 2 
        assert z6.dual == 2
        assert z7.real == 8.5
        assert z7.dual == 5
        with pytest.raises(TypeError):
            z1 + '1'
            '1' + z1
            True + z2
            z2 + True
        

    def test_multiplication(self):
        z1 = DualNumber(1,2)
        z2 = DualNumber(4,5)
        z3 = z1 * z2
        z4 = z1 * 2
        z5 = z2 * 1.5
        z6 = 2 * z1
        z7 = 1.5 * z2
        assert z3.real == 4
        assert z3.dual == 13
        assert z4.real == 2 
        assert z4.dual == 4
        assert z5.real == 6
        assert z5.dual == 7.5
        assert z6.real == 2 
        assert z6.dual == 4
        assert z7.real == 6
        assert z7.dual == 7.5
        with pytest.raises(TypeError):
            z1 * '1'
            '1' * z1
            True * z2
            z2 * False

    def test_reflective_operators(self):
        z = DualNumber(1,2)
        z1 = 1 + z
        z2 = z + 1
        assert z1.real == z2.real and z1.dual == z2.dual
        z3 = 2 * z
        z4 = z * 2
        assert z3.real == z4.real and z3.dual == z4.dual
        z5 = DualNumber(3,4)
        z6 = z+z5
        z7 = z5+z
        assert z6.real == z7.real and z6.dual == z7.dual
        z8 = z*z5
        z9 = z5*z
        assert z8.real == z9.real and z8.dual == z9.dual

    def test_subtract(self):
        z1 = DualNumber(1,2)
        z2 = DualNumber(7,3)
        z3 = z2-z1
        assert z3.real == 6
        assert z3.dual == 1
        z4 = z1-6
        assert z4.real == -5
        assert z4.dual == 2
        z5 = 6 - z1
        assert z5.real == 5
        assert z5.dual == -2
        z6 = z1-0.5
        assert z6.real == 0.5
        assert z6.dual == 2
        z7 = 2.5-z1
        assert z7.real == 1.5
        assert z7.dual == -2
        with pytest.raises(TypeError):
            z1 - '1'
            '1' - z1
            True - z2
            z2 - True

    def test_division(self):
        z1 = DualNumber(1,2)
        z2 = DualNumber(6,6)
        z3 = z2/z1
        assert z3.real == 6
        assert z3.dual == -6
        z4 = z1/2
        assert z4.real == .5
        assert z4.dual == 1
        z5 = 6 / z1
        assert z5.real == 6
        assert z5.dual == -12
        z6 = z1/0.5
        assert z6.real == 2
        assert z6.dual == 4
        z7 = 2.5/z1
        assert z7.real == 2.5
        assert z7.dual == -5
        with pytest.raises(TypeError):
            z1 / '1'
            z1.__rtruediv__("1")
            True / z2
            z2 / True

    def test_pow(self):
        z = DualNumber(1,2)
        z1 = 2**z
        assert z1.real == 2 and z1.dual == 4 * np.log(2)
        z2 = DualNumber(2,3)
        z3 = z.__pow__(z2)
        assert z3.real == 1 and z3.dual == 4
        with pytest.raises(TypeError):
            "1" ** z
            z.__pow__("1")