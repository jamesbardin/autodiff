#!/usr/bin/env python3
# File       : test_func.py
# Description: Test cases for Func class
# Copyright 2022 Harvard University. All Rights Reserved


import pytest
import numpy as np
import autodiff as ad
from autodiff.dualnumber import DualNumber
from autodiff.func import Func

class TestFunc:
    '''Test class for Func constructor and methods'''

    def test_init(self):
        '''Test for the Func construtor'''

        f = lambda x: x+1
        num_in = 1
        num_out = 1
        with pytest.raises(AssertionError):
            Func(f, 3.2, 1)
            Func(f, 3, 1.8)
            Func(f, 0, 5)
            Func(f, 8, 0)
        
        my_function = Func(f, num_in, num_out)
        assert my_function.func == f
        assert my_function.num_inputs == num_in
        assert my_function.num_outputs == num_out
    
    def test_repr(self):
        '''Test for repr of Func class instances'''

        f = lambda x: x+1
        num_in = 1
        num_out = 1
        my_function = Func(f, num_in, num_out)
        assert repr(my_function) == f"Function {f} with {num_in} input(s) and {num_out} output(s)"

    def test_jacobian(self):
        '''Test for jacobian generating method of Func class instances'''

        f = lambda x: x+1
        def ff(x):
            return x+1, x+4
        num_in = 1
        num_out = 1
        my_function = Func(f, num_in, num_out)
        my_function2 = Func(f, num_in, num_out + 1)
        my_function3 = Func(ff, num_in, num_out)

        with pytest.raises(AssertionError):
            my_function.jacobian(point = 'bana')
            my_function2.jacobian(point = [6])
            my_function3.jacobian(point = [6])
        with pytest.raises(ValueError):
            my_function.jacobian(point = [6,5])

        assert my_function.jacobian(6) == 1
        assert my_function.jacobian(6, reverse=True) == 1

        def fff(x,y):
            return y * x**2, 5 * x + ad.sin(y)
        my_function4 = Func(fff, 2, 2)
        assert np.array_equal(my_function4.jacobian([2.5,np.pi]), np.array([[2*2.5*np.pi, 2.5**2],[5, np.cos(np.pi)]]))
        assert np.array_equal(my_function4.jacobian([2.5,np.pi], reverse=True), np.array([[2*2.5*np.pi, 2.5**2],[5, np.cos(np.pi)]]))

    def test_eval(self):
        '''Test for function and derivative evaluation method of Func class instances'''

        f = lambda x: x+1
        def ff(x):
            return 3*x + 1, x**2 + 4
        num_in = 1
        num_out = 1
        my_function = Func(f, num_in, num_out)
        my_function2 = Func(f, num_in, num_out+1)
        my_function3 = Func(ff, num_in, num_out)
        
        with pytest.raises(AssertionError):
            my_function.eval(point = 'bana', seed_vector = [4])
            my_function.eval(point = np.array([4]), seed_vector = 9)
            my_function2.eval([2.5],[1])
            my_function3.eval([2.5],[1])
        with pytest.raises(ValueError):
            my_function.eval(point = [6], seed_vector = [4,2])
        with pytest.raises(ValueError):
            my_function.eval(point = [6,2], seed_vector = [4])
            my_function.eval(point = [6,3], seed_vector = [4,2])
        
        fff = lambda x: ad.exp(x**2) + 1
        my_function4 = Func(fff, num_in, num_out)
        out_val, out_deriv = my_function4.eval(3,[1])
        assert np.array_equal(out_val,np.array([fff(3)]))
        assert np.array_equal(out_deriv,np.array([2*3*np.exp(3**2) * 1]))
        
        def ffff(x,y,z):
            return z * y * x**2 + ad.cos(x), 5 * x + y - z 
        
        my_function5 = Func(ffff, 3, 2)
        out_val, out_deriv = my_function5.eval([3,2,4],[1,0,0])
        assert np.array_equal(out_val, np.array(ffff(3,2,4)))
        assert np.array_equal(out_deriv, np.array([4*2*2*3 - np.sin(3), 5]))



    


