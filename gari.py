import numpy
import functools
import operator

def img2chromosome(img_arr):
    """
    Represents the image as a 1D vector.
    
    img_arr: The image to be converted into a vector.
    
    Returns the vector.
    """

    return numpy.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))

def chromosome2img(vector, shape):
    """
    Converts a 1D vector into an array.
    
    vector: The vector to be converted into an array.
    shape: The shape of the target array.
    
    Returns the array.
    """

    # Check if the vector can be reshaped according to the specified shape.
    if len(vector) != functools.reduce(operator.mul, shape):
        raise ValueError("A vector of length {vector_length} into an array of shape {shape}.".format(vector_length=len(vector), shape=shape))

    return numpy.reshape(a=vector, newshape=shape)
