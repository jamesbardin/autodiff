#!/usr/bin/env python3
# File       : dualnumber.py
# Description: Dual number with support for operators
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np

class DualNumber:
    '''DualNumber class with overloaded operators for forward mode AD.'''

    _supported_scalars = (int, float)

    def __init__(self, real, dual = 1):
        '''Class constructor for a DualNumber object.

        Parameters
        ----------
        real : int, float
            The real component of the dual number used for evaluating the primal trace
        
        dual : int, float
            The dual component of the dual number, used for evaluating tangent trace, defaulted to 1.

        Attributes
        ----------
            real : int, float
                The real component of the dual number used for evaluating the primal trace
            dual : int, float
                The dual component of the dual number, used for evaluating tangent trace, defaulted to 1.
        Raises
        -------
        TypeError : raise TypeError if real or dual components are not of int, float type
        '''
        if not isinstance(real, self._supported_scalars):
            raise TypeError(f"Unsupported type {type(real)}")
        if not isinstance(dual, self._supported_scalars):
            raise TypeError(f"Unsupported type {type(dual)}")
        self.real = real
        self.dual = dual
    
    def __repr__(self):
        '''Prints dual number representation

        Returns
        -------
        string
            Returns a string specifying the dual number with its real component and dual component
            in scientific notation.
        '''
        return f"DualNumber(real: {self.real:e}, dual: {self.dual:e})"

    def __eq__(self, other):
        '''Assess equality between dual numbers

        Parameters
        -------
        other : DualNumber
            The other DualNumber with which equality is being assessed

        Returns
        -------
        bool
            Returns a boolean that is true if both the real and dual components of each
            dual number are equal, and false otherwise.

        Raises
        -------
        TypeError : raise TypeError if comparison object is not of DualNumber type
        '''
        if not isinstance(other, DualNumber):
            raise TypeError(f"Unsupported type {type(other)}")
        else:
            return self.real == other.real and self.dual == other.dual
    
    def __ne__(self, other):
        '''Assess inequality between dual numbers

        Parameters
        -------
        other : DualNumber
            The other DualNumber with which inequality is being assessed

        Returns
        -------
        bool
            Returns a boolean that is true if the real or dual components of each
            dual number are not equal, and false otherwise.

        Raises
        -------
        TypeError : raise TypeError if comparison object is not of DualNumber type
        '''
        return not self.__eq__(other)

    def __neg__(self):
        '''Unary negation of dual number

        Returns
        -------
        DualNumber
            Returns a dual number with real and dual components that have been negated.
        '''
        return DualNumber(-self.real, -self.dual)
    
    def __add__(self, other):
        '''Addition for dual numbers with ints, floats, and other dual numbers

        Parameters
        -------
        other : DualNumber, int, float
            The other DualNumber, int, or float being added to a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the addition performed. Integer and
            float addition is supported for adding to DualNumber as well as DualNumber
            addition with component-wise addition.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type {type(other)}")
        if isinstance(other, self._supported_scalars):
            return DualNumber(other+self.real, self.dual)
        else:
            return DualNumber(self.real+other.real, self.dual+other.dual)

    def __sub__(self, other):
        '''Subtraction for dual numbers with ints, floats, and other dual numbers

        Parameters
        -------
        other : DualNumber, int, float
            The other DualNumber, int, or float being subtracted from a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the subtraction performed. Integer and
            float subtraction is supported for subtracting from a DualNumber as well as DualNumber
            subtraction with component-wise subtraction.

        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        return self.__add__(other*(-1))

    def __mul__(self, other):
        '''Multiplication for dual numbers with ints, floats, and other dual numbers

        Parameters
        -------
        other : DualNumber, int, float
            The other DualNumber, int, or float being multiplied with a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the multiplication performed. Integer and
            float multiplication is supported for multiplying with a DualNumber as well as the product 
            of two DualNumber objects.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type {type(other)}")
        if isinstance(other, self._supported_scalars):
            return DualNumber(other*self.real, other*self.dual)
        else:
            return DualNumber(self.real*other.real, self.real*other.dual+self.dual*other.real)

    def __truediv__(self, other):
        '''Division for dual numbers with ints, floats, and other dual numbers

        Parameters
        -------
        other : DualNumber, int, float
            The other DualNumber, int, or float dividing a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the division performed. Integer and
            float division is supported for dividing a DualNumber as well as the division 
            of two DualNumber objects.

        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type {type(other)}")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real/other, self.dual/other)
        else:
            return DualNumber(self.real/other.real, (self.dual*other.real-self.real*other.dual)/other.real**2)

    def __pow__(self, other):
        '''Power for dual numbers with ints, floats, and other dual numbers

        Parameters
        -------
        other : DualNumber, int, float
            The other DualNumber, int, or float that the DualNumber object is raised to the power of

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the power performed. Integer and
            float exponents is supported for exponentiating a DualNumber as well as the power 
            of a DualNumber to a DualNumber.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type {type(other)}")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real**other, other * pow(self.real, other-1)*self.dual)
        else:
            return DualNumber(self.real**other.real, self.real**other.real *(other.dual*np.log(self.real)+self.dual*other.real/self.real))

    def __radd__(self, other):
        '''Reverse addition for adding DualNumber to int or float

        Parameters
        -------
        other : int, float
            The other int or float being added with a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the addition performed. Integer and
            float addition is supported and implemented via commutativity

        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        return self.__add__(other)
    
    def __rsub__(self, other):
        '''Reverse subtraction for subtracting DualNumber from int or float

        Parameters
        -------
        other : int, float
            The other int or float being subtracted by a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the subtraction performed. Integer and
            float subtraction is supported.

        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        return (-1*self).__add__(other)

    def __rmul__(self, other):
        '''Reverse multiplication for multiplying DualNumber by int or float

        Parameters
        -------
        other : int, float
            The other int or float being multiplied with a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the multiplication performed. Integer and
            float multiplication is supported and implemented via commutativity

        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        '''Reverse division for dividing int or float by DualNumber

        Parameters
        -------
        other : int, float
            The other int or float being divided by a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the division performed. Integer and
            float division is supported.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        if not isinstance(other, (self._supported_scalars)):
            raise TypeError(f"Unsupported type {type(other)}")
        else:
            return DualNumber(other/self.real, -other*self.dual/(self.real**2))

    def __rpow__(self, other):
        '''Reverse power for exponentiating int or float by DualNumber

        Parameters
        -------
        other : int, float
            The other int or float being expoentiated by a DualNumber object

        Returns
        -------
        DualNumber
            Returns a new DualNumber object with the power performed. Integer and
            float power exponentiation is supported.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''
        if not isinstance(other, (self._supported_scalars)):
            raise TypeError(f"Unsupported type {type(other)}")
        else:
            return DualNumber(other**self.real, other**self.real * self.dual * np.log(other))
