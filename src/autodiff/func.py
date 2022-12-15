#!/usr/bin/env python3
# File       : func.py
# Description: Function class
# Copyright 2022 Harvard University. All Rights Reserved

'''Function Module with Func class definition'''


import numpy as np
import graphviz
from graphviz import Digraph
from .dualnumber import DualNumber
from .rNode import rNode

class Func:
    '''Func class to be used for forward mode AD evaluation of derivatives.'''


    def __init__(self, function, num_inputs, num_outputs):
        '''Class constructor for a Func object.
        Parameters
        ----------
        function : function object
            The user-defined mathematical function that users wish to differentiate.
        
        num_inputs : int
            The number of inputs to the function users wish to differentiate.
        num_outputs : int
            The number of outputs to the function users wish to differentiate.
        Attributes
        ----------
            func : function object
                The user-defined mathematical function that users wish to differentiate.
            num_inputs : int
                The number of inputs to the function users wish to differentiate.
            num_outputs : int
                The number of outputs to the function users wish to differentiate.
        '''
        assert isinstance(num_inputs, int), 'num_inputs must be an integer'
        assert isinstance(num_outputs, int), 'num_outputs must be an integer'
        assert num_inputs > 0, 'num_inputs must greater than 0'
        assert num_outputs > 0, 'num_outputs must greater than 0'

        self.func = function
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def __repr__(self):
        return f"Function {self.func} with {self.num_inputs} input(s) and {self.num_outputs} output(s)"

    def __call__(self, point):
        '''The function evaluation at a given point.
        Parameters
        ----------
        point : int,float,list, np.ndarray
            The input point at wish users wish to evaluate the function's jacobian matrix. Can be int/float if & only if input dimension = 1.
        
        Returns
        -------
        When the evaluation is a scalar:
            int, float
                The scalar value representing the evaluation of the function at the inputted point
        Otherwise:
            np.array
                The num_outputs array representing evaluation for the function at the inputted point
        Examples
        --------
        
        >>> fofx = lambda x : 4 * x**3 - 3
        >>> f = Func(fofx, 1, 1)
        >>> p = [2]
        >>> f(p)
        29
        >>> def fofx(x,y):
                return y * x**2, (x-y)
        >>> f = Func(fofx,2,2)
        >>> p = [2,3]
        >>> f(p)
        array([12,  -1])
        '''
        if isinstance(point,(int,float)):
            point = [point]
        assert isinstance(point, (list,np.ndarray)), 'point must be a int, float, list, or numpy array'
        if len(point) != self.num_inputs:
            raise ValueError(f"Dimension mismatch between inputted point and function number of input")
        if self.num_outputs == 1:
            return self.func(*point)
        else:
            return np.array(self.func(*point))

    def _forward(self, point):
        jacobian = np.zeros((self.num_outputs,self.num_inputs))
        identity = np.identity(self.num_inputs)
        for i in range(self.num_inputs):
            seed_vector = identity[:,i]
            zs = []
            for j in range(self.num_inputs):
                assert isinstance(point[j], (int,float)), 'point input must consist of integers or floats only'
                
                zs.append(DualNumber(point[j],seed_vector[j]))
            
            res = self.func(*zs)

            if isinstance(res, DualNumber):
                assert  self.num_outputs == 1, 'Dimension mismatch between function output and specified number of outputs'
                jacobian[:,i] = res.dual
            else:
                assert  len(res) == self.num_outputs, 'Dimension mismatch between function output and specified number of outputs'
                for k in range(self.num_outputs):
                    jacobian[:,i][k] = res[k].dual
        return jacobian


    def _reverse(self, point):
        rNode.increment = 0      
        jacobian = np.zeros((self.num_outputs,self.num_inputs))      

        zs = []
        for p in point:
            assert isinstance(p, (int,float)), 'point input must consist of integers or floats only'
            zs.append(rNode(p))
    
        res = self.func(*zs)
        if isinstance(res, rNode):
            assert  self.num_outputs == 1, 'Dimension mismatch between function output and specified number of outputs'
            jacobian[0,:] = res.backward(self.num_inputs)
        else:
            assert  len(res) == self.num_outputs, 'Dimension mismatch between function output and specified number of outputs'
            for i, node in enumerate(res):
                jacobian[i,:] = node.backward(self.num_inputs)
        return jacobian

    def jacobian(self, point, reverse = False):
        '''The jacobian matrix of a function at a given point.
        Parameters
        ----------
        point : int,float,list, np.ndarray
            The input point at wish users wish to evaluate the function's jacobian matrix. Can be int/float if & only if input dimension = 1.
        
        reverse : bool
            Set to true if users wish to evaluate the function's jacobian using reverse mode automatic.  Default is False.
        Returns
        -------
        When the jacobian is a scalar:
            int, float
                The scalar value representing the jacobian of the function at the inputted point
        Otherwise:
            np.ndarray
                The num_outputs by num_inputs jacobian matrix for the function at the inputted point
        Examples
        --------
        
        >>> fofx = lambda x : 4 * x**3 - 3
        >>> f = Func(fofx, 1, 1)
        >>> p = [2]
        >>> f.jacobian(p)
        48.0
        >>> def fofx(x,y):
                return y * x**2, (x-y)
        >>> f = Func(fofx,2,2)
        >>> p = [2,3]
        >>> f.jacobian(p)
        array([[12.,  4.],
              [ 1., -1.]])
        '''
        if isinstance(point,(int,float)):
            point = [point]
        assert isinstance(point, (list,np.ndarray)), 'point must be a int, float, list, or numpy array'
        assert reverse == True or reverse == False, 'reverse must be True or False'
        if len(point) != self.num_inputs:
            raise ValueError(f"Dimension mismatch between inputted point and function number of input")
        if not reverse:
            jacobian = self._forward(point)
        else:
            jacobian = self._reverse(point)

        if jacobian.shape == (1,1):
            return jacobian[0,0]
        else:
            return jacobian

    def eval(self, point, seed_vector):
        '''Evaluate the function value and derivative in a given direction at a given point.
        Parameters
        ----------
        point : int, float, list, np.ndarray
            The input point at wish users wish to evaluate the function and the function's derivative. Can be int/float if & only if input dimension = 1.
        seed_vector : list, np.ndarray
            The direction in wish users wish to evaluate the function's derivative     
        Returns
        -------
        tuple
            The first element is an np.ndarray with num_outputs length representing the function's value
            evaluated at the given point.  The second element is an np.ndarray with num_outputs length
            representing the value of the derivative at the given point in the given direction.
        Examples
        --------
        
        >>> fofx = lambda x : 2 * x**2 + 4
        >>> f = Func(fofx, 1, 1)
        >>> p = [4]
        >>> direction = [1]
        >>> f.eval(p, direction)
        (array([36.]), array([16.]))
        >>> def fofx(x, y):
                return y + x**2, x - 3*y
        >>> f = Func(fofx,2,2)
        >>> p = [4,2]
        >>> direction = [1,0]
        >>> f.eval(p, direction)
        (array([18., -2.]), array([8., 1.]))
        '''

        if isinstance(point,(int,float)):
            point = [point]
        assert isinstance(point, (list,np.ndarray)), 'point must be a int, float, list or numpy array'
        assert isinstance(seed_vector, (list,np.ndarray)), 'seed_vector must be a list or numpy array'

        if len(point) != self.num_inputs:
            raise ValueError(f"Dimension mismatch between inputted point and function number of input")
        
        elif len(point) != len(seed_vector):
            raise ValueError(f"Dimension mismatch between inputted point and seed vector")

        else:
            zs = []
            for j in range(self.num_inputs):
                zs.append(DualNumber(point[j],seed_vector[j]))
                
            res = self.func(*zs)
            val = np.zeros(self.num_outputs)
            deriv = np.zeros(self.num_outputs)

            if isinstance(res, DualNumber):
                assert  self.num_outputs == 1, 'Dimension mismatch between function output and specified number of outputs'
                val[0] = res.real
                deriv[0] = res.dual
            else:
                assert  len(res) == self.num_outputs, 'Dimension mismatch between function output and specified number of outputs'
                for k in range(self.num_outputs):
                    val[k] = res[k].real
                    deriv[k] = res[k].dual
            
            return val, deriv
    
    
    @staticmethod
    def _display_graph(graph_object , graph_direction):
        
        nodes, edges = graph_object    
        dot = Digraph(format='svg', graph_attr={'rankdir': graph_direction})    
        for n in nodes:
            dot.node(name=str(id(n)), label = f" val: {n.val:.3f} ", shape='record')
            if n.oper:
                dot.node(name=str(id(n)) + n.oper, label=n.oper)
                dot.edge(str(id(n)) + n.oper, str(id(n)))    
        for n1, n2 in edges: 
            dot.edge(str(id(n1)), str(id(n2)) + n2.oper)    
        return dot

    
    def graph(self, point, graph_direction = 'LR'):
        '''Display the computational graph of the function
        
        Parameters
        ----------
        point : int,float,list, np.ndarray
            The input point at wish users wish to calculate the function's computational graph.
        
        graph_direction : str
            A string indicating if users would like the computational graph display 'LR', left-to-right,
            or 'TB' top-to-bottom. Defaults to 'LR', left-to-right
        
        Returns
        -------
            graphviz.graphs.Digraph
                A graphviz.graphs.Digraph object that displays the computational graph of the function

        Examples
        --------
        
        >>> def fofx(x,y,w):
            return  w + x**(2-y+w) + 2*y
        >>> f = Func(fofx,3,1)
        >>> p = [5,2.09,4.8]
        >>> f.graph(p)
        {computational graph displayed from left to right}
        >>> f.graph(p,'TB')
        {computational graph displayed from top to bottom}
        '''
        
        
        
        assert self.num_outputs == 1, 'computational graph only available for functions of Rn -> R1 and inputted output dimension > 1'
        assert graph_direction in ['LR', 'TB'], "graph direction must be either 'LR' for left-to-right or 'TB' for top-to-bottom" 
        
        if isinstance(point,(int,float)):
            point = [point]
        assert isinstance(point, (list,np.ndarray)), 'point must be a int, float, list, or numpy array'
        
        if len(point) != self.num_inputs:
            raise ValueError(f"Dimension mismatch between inputted point and function number of input")
        
        zs = []
        for p in point:
            assert isinstance(p, (int,float)), 'point input must consist of integers or floats only'
            zs.append(rNode(p))

        res = self.func(*zs)
        assert isinstance(res, rNode), 'computational graph only available for functions of Rn -> R1 and inputted function has >1 outputs'
        
        graph_object = res.graph_object()
        
        return self._display_graph(graph_object,graph_direction)