#!/usr/bin/env python3
# File       : rNode.py
# Description: Reverse Mode Nodes
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np

# ### need to add checks for supported types, docstrings, tests, etc.

class rNode:
    '''rNode class with overloaded operators and graph compatibility for reverse mode AD.
    
    Attributes
    ----------    
    increment : int
        Increment to keep track of how many instances of the rNode
        have been created before the jacobian is found

    '''
    increment = 0
    
    def __init__(self, val, parents = (), oper = ''):
        '''Class constructor for a rNode object.
        Parameters
        ----------
        val : int, float
            The value of the node
        
        parents : tuple, empty or of rNode objects
            The parent nodes of a given node
        
        oper : str
            The operation that lead to the creation of the node
        
        Attributes
        ----------
        val : int, float
            The value of the node

        der : int, float
            The value of the adjoint of the node

        backward_func : function
            Function defining how to update the adjoint value as you traverse backwards
            through the graph
        
        oper : str
            The operation that lead to the creation of the node

        parents : set, empty or of rNode objects
            The parent nodes of a given node

        name : str
            The name representing the node in the computational graph            

        '''
        self.val = val
        self.der = 0
        self.backward_func = lambda: None
        self.oper = oper
        self.parents = set(parents)
        self.name = 'v' + str(rNode.increment)
        rNode.increment += 1
    
    def __repr__(self):
        '''Prints rNode representation
        
        Returns
        -------
        string
            Returns a string specifying the node value, the adjoint value, the node name,
            the operation that created that node, and how many parents the node has.
        '''
        return f"rNode(value: {self.val:e}, derivative: {self.der:e},  name= {self.name}, op={self.oper}, num_parents={len(self.parents)})"
    
    def __neg__(self):
        '''Unary negation of the node
        Returns
        -------
        rNode
            Returns an rNode with a value that has been negated
        '''
        return self * -1

    def __add__(self, other):
        '''Addition for rNodes with ints, floats, and other rNodes
        
        Parameters
        -------
        other : rNodes, int, float
            The other rNode, int, or float being added to a rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the addition performed and the correct
            backwards func, parents, and operation. Integer and float addition is
            supported for adding to rNode as well as rNode to rNode addition.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''

        if not isinstance(other, (int,float, rNode)):
            raise TypeError(f"Unsupported type {type(other)}")
        
        if isinstance(other, (int,float)):
            other = rNode(other)
        
        res = rNode(self.val + other.val, (self, other), '+')

        def _backward_fn():
            self.der += 1 * res.der
            other.der += 1 * res.der
        
        res.backward_func = _backward_fn

        return res
    

    def __radd__(self, other):
        '''Reverse addition for adding rNode to int or float
        
        Parameters
        -------
        other : int, float
            The other int or float being added with a rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the addition performed and the correct
            backwards func, parents, and operation.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''

        return self + other

    def __sub__(self, other):
        '''Subtraction for rNodes with ints, floats, and other rNodes
        
        Parameters
        -------
        other : rNodes, int, float
            The other rNode, int, or float being subtracted from a rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the subtraction performed and the correct
            backwards func, parents, and operation. Integer and float subtraction is
            as well as rNode from rNode subtraction.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''
        
        if not isinstance(other, (int,float, rNode)):
            raise TypeError(f"Unsupported type {type(other)}")
        
        if isinstance(other, (int,float)):
            other = rNode(other)
        
        res = rNode(self.val - other.val, (self, other), '-')

        def _backward_fn():
            self.der  += 1 * res.der
            other.der += -1 * res.der
        
        res.backward_func = _backward_fn

        return res
    
    def __rsub__(self, other):
        '''Reverse subtraction for subtracting rNode from int or float
        
        Parameters
        -------
        other : int, float
            The other int or float being subtracted by a rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the subtraction performed and the correct
            backwards func, parents, and operation. 
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or DualNumber type
        '''

        return other + (-self)

    def __mul__(self, other):
        '''Multiplication for rNodes with ints, floats, and other rNodes
        
        Parameters
        -------
        other : rNodes, int, float
            The other rNode, int, or float being multiplied by an rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the multiplication performed and the correct
            backwards func, parents, and operation. Integer and float multiplication is
            supported as well as rNode to rNode mulitplication.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''


        if not isinstance(other, (int,float, rNode)):
            raise TypeError(f"Unsupported type {type(other)}")

        if isinstance(other, (int,float)):
            other = rNode(other)

        res = rNode(self.val * other.val, (self, other), 'x')

        def _backward_fn():
            self.der  += other.val * res.der
            other.der += self.val * res.der
        
        res.backward_func = _backward_fn

        return res
    
    def __rmul__(self, other):
        '''Reverse multiplication for multiplying rNode by int or float
        
        Parameters
        -------
        other : int, float
            The other int or float being multiplied with a rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the multiplication performed and the correct
            backwards func, parents, and operation.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''

        return self * other
    
    def __pow__(self, other):
        '''Power for rNodes with ints, floats, and other rNodes
        
        Parameters
        -------
        other : rNode, int, float
            The other rNode, int, or float that the rNode object is raised to the power of
        
        Returns
        -------
        rNode
            Returns a new rNode object with the power performed and the correct
            backwards func, parents, and operation. Integer and float exponents
            are supported for exponentiating a rNode as well as the power 
            of a rNode to a rNode.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''
        if not isinstance(other, (int,float, rNode)):
            raise TypeError(f"Unsupported type {type(other)}")
        
        if isinstance(other, (int, float)):
            other = rNode(other)
        
        res = rNode(self.val**other.val, (self, other), f'^{other.val}')

        def _backward_fn():
            self.der += (other.val * self.val**(other.val-1)) * res.der
            other.der += (self.val**other.val  * np.log(self.val)) * res.der

        res.backward_func = _backward_fn

        return res
            
    def __rpow__(self, other):
        '''Reverse power for exponentiating int or float by rNode
       
        Parameters
        -------
        other : int, float
            The other int or float being expoentiated by an rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the power performed and the correct
            backwards func, parents, and operation. Integer and float power
            exponentiation is supported.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int or float type
        '''

        if not isinstance(other, (int, float)):
            raise TypeError(f"Unsupported type {type(other)}")
        res = rNode(other**self.val, (self,), f'{other}^')

        def _backward_fn():
            self.der += (other**self.val * np.log(other)) * res.der

        res.backward_func = _backward_fn

        return res

    def __truediv__(self, other):
        '''Division for rNodes with ints, floats, and other rNodes
        
        Parameters
        -------
        other : rNode, int, float
            The other rNode, int, or float dividing a rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the division performed and the correct
            backwards func, parents, and operation. Integer and float division is
            supported as well as the division of two rNode objects.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''

        return self * other**-1

    def __rtruediv__(self, other):
        '''Reverse division for dividing int or float by rNode
        
        Parameters
        -------
        other : int, float
            The other int or float being divided by a rNode object
        
        Returns
        -------
        rNode
            Returns a new rNode object with the division performed and the correct
            backwards func, parents, and operation. Integer and float division is
            supported.
        
        Raises
        -------
        TypeError : raise TypeError if other object is not of int, float, or rNode type
        '''

        return other * self**-1
    
    def _sort_nodes(self):
        '''Topological sort of the nodes in the function's computational graph after forward pass
        
        Returns
        -------
        list
            Returns a list of rNode object representative of the topological
            ordering of the nodes in the function's computational graph. 
        '''
        topo_sorted = []
        visited_nodes = set()

        def toposort(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for parent_node in node.parents:
                    toposort(parent_node)
                topo_sorted.append(node)
        toposort(self)
        return topo_sorted


    def graph_object(self):
        '''Create list of nodes and edges used to construct computational graph
        
        Returns
        -------
        tuple
            Returns a tuple of two sets. The first set is a set of all the nodes
            in the computational graph. The second set is a set of tuples where each tuple
            has rNode objects as entries, and represent rNodes that should have an edge
            between them in the computational graph.
        '''
        nodes, edges = set(), set()
        def build(node):
            if node not in nodes:
                nodes.add(node)
                for parent in node.parents:
                    edges.add((parent, node))
                    build(parent)
        
        build(self)
        return nodes, edges

    def backward(self, num_inputs):
        '''Reverse pass to accumulative gradients

        Parameters
        -------
        num_inputs : int
            The number of inputs the function has
        
        Returns
        -------
        np.ndarray
            Returns a new np.ndarray with each entry corresponding to the derivative
            of self node with respect one of the function inputs, such that, if this
            node is a function output, the array itself corresponds to one row of
            the jacobian matrix

        '''
        topo_order = self._sort_nodes()
        
        #zero the adjoints after each backwards pass
        for node in topo_order:
            node.der = 0

        self.der = 1
        
        jacobian_row = np.zeros(num_inputs)

        for node in reversed(topo_order):
            if int(node.name[1:]) < num_inputs:
                jacobian_row[int(node.name[1:])] = node.der
            node.backward_func()
        
        return jacobian_row
