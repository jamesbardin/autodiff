#!/usr/bin/env python3
# File       : operators.py
# Description: operators implemented for dual numbers
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np
from .dualnumber import DualNumber
from .rNode import rNode

_supported_scalars = (int, float)

def sin(x):
    '''Sine operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float the sine is evaluated on 

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the sine operation performed. Integer and
        floats are supported, along with the sine of dual numbers through application 
        of the chain rule.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber, rNode)):
        raise TypeError(f"Unsupported type {type(x)}")
    elif isinstance(x, _supported_scalars):
        return np.sin(x)
    elif isinstance(x, rNode):
        res = rNode(np.sin(x.val), (x,), 'sin')
        def _backward_fn():
            x.der += np.cos(x.val)*res.der
        res.backward_func = _backward_fn
        return res
    else:
        return DualNumber(np.sin(x.real), x.dual*np.cos(x.real))

def cos(x):
    '''Cosine operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float the cosine is evaluated on 

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the cosine operation performed. Integer and
        floats are supported, along with the cosine of dual numbers through application 
        of the chain rule.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber, rNode)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return np.cos(x)
    elif isinstance(x, rNode):
        res = rNode(np.cos(x.val), (x,), 'cos')
        def _backward_fn():
            x.der += -np.sin(x.val)*res.der
        res.backward_func = _backward_fn
        return res
    else:
        return DualNumber(np.cos(x.real), -x.dual*np.sin(x.real))

def tan(x):
    '''Tangent operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float the tangent is evaluated on 

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the tangent operation performed. Integer and
        floats are supported, along with the tangent of dual numbers through application 
        of the chain rule.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber, rNode)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return np.tan(x)
    elif isinstance(x, rNode):
        res = rNode(np.tan(x.val), (x,), 'tan')
        def _backward_fn():
            x.der += 1/(np.cos(x.val)**2)*res.der
        res.backward_func = _backward_fn
        return res
    else:
        return DualNumber(np.tan(x.real),x.dual/(np.cos(x.real)**2))

def exp(x):
    '''Exp operation for dual numbers and ints or floats (base e)

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float being exponentiated

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the exp operation performed. Integer and
        floats are supported, along with the exponent of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return np.exp(x)
    else:
        return DualNumber(np.exp(x.real),x.dual*np.exp(x.real))

def log(x):
    '''Log operation for dual numbers and ints or floats (base e)

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the natural logarithm

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the logarithm operation performed. Integer and
        floats are supported, along with the natural logarithm of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    ValueError : raise ValueError if argument is nonpositive
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        if x <= 0:
            raise ValueError("Logarithm undefined on input")
        return np.log(x)
    else:
        if x.real <= 0:
            raise ValueError("Logarithm undefined on input")
        return DualNumber(np.log(x.real), x.dual/x.real)

def log_base(x, base):
    '''Log base operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float argument for the logarithm
    x : int, float
        The int, or float base of the logarithm

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the logarithm operation performed. Integer and
        floats are supported, along with the logarithm of dual numbers of any int/float base.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    ValueError : raise ValueError if argument is nonpositive
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        if x <= 0:
            raise ValueError("Logarithm undefined on input")
        # Applying change of base formula
        return np.log(x)/np.log(base)
    else:
        if x.real <= 0:
            raise ValueError("Logarithm undefined on input")
        return DualNumber(np.log(x.real)/np.log(base), x.dual/(np.log(base)*x.real))


def arcsin(x):
    '''Arcsin operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the arcsin operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the arcsin operation performed. Integer and
        floats are supported, along with the arcsin of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    ValueError : raise ValueError if argument is not between -1 and 1 inclusive
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        if x < -1 or x > 1:
            raise ValueError("Invalid domain of operator")
        return np.arcsin(x)
    else:
        if x.real < -1 or x.real > 1:
            raise ValueError("Invalid domain of operator")
        return DualNumber(np.arcsin(x.real), x.dual/np.sqrt(1-x.real**2))

def arccos(x):
    '''Arccos operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the arccos operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the arccos operation performed. Integer and
        floats are supported, along with the arccos of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    ValueError : raise ValueError if argument is not between -1 and 1 inclusive
    '''
    return -arcsin(x)

def arctan(x):
    '''Arctan operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the arctan operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the arctan operation performed. Integer and
        floats are supported, along with the arctan of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return np.arctan(x)
    else:
        return DualNumber(np.arctan(x.real), x.dual/(1+x.real**2))

def sinh(x):
    '''Sinh operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the sinh operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the sinh operation performed. Integer and
        floats are supported, along with the sinh of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return np.sinh(x)
    else:
        return DualNumber(np.sinh(x.real), x.dual*np.cosh(x.real))

def cosh(x):
    '''Cosh operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the cosh operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the cosh operation performed. Integer and
        floats are supported, along with the cosh of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return np.cosh(x)
    else:
        return DualNumber(np.cosh(x.real), x.dual*np.sinh(x.real))

def tanh(x):
    '''Tanh operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the tanh operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the tanh operation performed. Integer and
        floats are supported, along with the tanh of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return np.tanh(x)
    else:
        return DualNumber(np.cosh(x.real), x.dual/(np.cosh(x.real)**2))

def logistic(x):
    '''Logistic operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the tanh operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the tanh operation performed. Integer and
        floats are supported, along with the tanh of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        return 1/(1+exp(-x))
    else:
        return DualNumber(1/(1+exp(-x.real)), x.dual*np.exp(x.real)/((np.exp(x.real)+1)**2))

def sqrt(x):
    '''Square root operation for dual numbers and ints or floats

    Parameters
    -------
    x : DualNumber, int, float
        The DualNumber, int, or float used for the sqrt operation

    Returns
    -------
    DualNumber
        Returns a new DualNumber object with the sqrt operation performed. Integer and
        floats are supported, along with the sqrt of dual numbers.

    Raises
    -------
    TypeError : raise TypeError if argument is not of int, float, or DualNumber type
    ValueError : raise ValueError if argument is negative
    '''
    if not isinstance(x, (*_supported_scalars, DualNumber)):
        raise TypeError(f"Unsupported type {type(x)}")
    if isinstance(x, _supported_scalars):
        if x < 0:
            raise ValueError("Invalid square root argument (must be >= 0)")
        return np.sqrt(x)
    else:
        if x.real < 0:
            raise ValueError("Invalid square root argument (must be >= 0)")
        return DualNumber(np.sqrt(x.real), x.dual/(2*np.sqrt(x.real)))