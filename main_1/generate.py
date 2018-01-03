import numpy as np
import matplotlib.pyplot as plt


class generate():
    """description of class"""

    def __init__(self):
        return

    def generate_cosine(self,n):
        """
        generate_cosine(self,n)
        returns x,t

        Parameters
        ------------------------
        n: Number of points

        Returns
        ------------------------
        x: np array
        t: np array

        Note
        ------------------------
        This generates points from a cosine and returns N-dimentional vectors x and t 
        where x contains evenly spaced values from 0 to 2*i and t_i from a normal
        distribution with mu cos(x_i) and standard deviation 0.2"""
        
        x = 2 * np.pi * np.linspace(0,1,n)
        t = np.random.normal(np.cos(x),0.2,n)
        return x, t

        


