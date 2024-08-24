'''
This is a module that initializes an entire 2D Lattice consisting of atoms with Spins.

This Method accepts a tuple of two integers, which specify the dimensions x and y
of the lattice.

The mean property of this class module can be called to return the total mean
of the atom spins.

Example usage:
    s = mcsim.Spins(n=(10, 10))
    where n is a tuple describing the dimensions

'''

import numbers
import matplotlib.pyplot as plt
import numpy as np

class Spins:
    """Field of spins on a two-dimensional lattice.

    Each spin is a three-dimensional vector s = (sx, sy, sz). Underlying data
    stucture (``self.array``) is a numpy array (``np.ndarray``) with shape
    ``(nx, ny, 3)``, where ``nx`` and ``ny`` are the number of spins in the x
    and y directions, respectively, and 3 to hold all three vector components of
    the spin.

    Parameters
    ----------
    n: Iterable

        Dimensions of a two-dimensional lattice ``n = (nx, ny)``, where ``nx``
        and ``ny`` are the number of atoms in x and y directions, respectively.
        Values of ``nx`` and ``ny`` must be positive integers.

    value: Iterable

        The value ``(sx, sy, sz)`` that is used to initialise all spins in the
        lattice. All elements of ``value`` must be real numbers. Defaults to
        ``(0, 0, 1)``.

    """

    def __init__(self, n, value=(0, 0, 1)):
        '''
        Parameters
        ----------
        n: Iterable

            Dimensions of a two-dimensional lattice ``n = (nx, ny)``, where ``nx``
            and ``ny`` are the number of atoms in x and y directions, respectively.
            Values of ``nx`` and ``ny`` must be positive integers.

        value: Iterable

            The value ``(sx, sy, sz)`` that is used to initialise all spins in the
            lattice. All elements of ``value`` must be real numbers. Defaults to
            ``(0, 0, 1)``.

        '''
        # Checks on input parameters.
        if len(n) != 2:
            raise ValueError(f"Length of iterable n must be 2, not {len(n)=}.")
        if any(i <= 0 or not isinstance(i, int) for i in n):
            raise ValueError("Elements of n must be positive integers.")

        if len(value) != 3:
            raise ValueError(f"Length of iterable value must be 3, not {len(n)=}.")
        if any(not isinstance(i, numbers.Real) for i in n):
            raise ValueError("Elements of value must be real numbers.")

        self.n = n
        self.array = np.empty((*self.n, 3), dtype=np.float64)
        self.array[..., :] = value

        if not np.isclose(value[0] ** 2 + value[1] ** 2 + value[2] ** 2, 1):
            # we ensure all spins' magnitudes are normalised to 1.
            self.normalise()

    @property
    def mean(self):
        """
        Approximate the mean of the spin directions of all atoms in the latice.
        Returns an array with the mean.
        """
        return np.mean(self.array, axis= (0,1))

    def __abs__(self):
        '''
        Returns the norm of each spin inside the 2 Dimensional latice
        '''
        euc = np.linalg.norm(self.array, axis =2)
        #I tried the axis in the testing playground and 2 was the correct one
        euc = euc.reshape((*self.n,1))
        #we are reshaping the array to fit what the pdf wants
        return euc

    def normalise(self):
        """Normalise the magnitude of all spins to 1."""
        self.array = self.array / abs(self) 
        # This computation will be failing until you implement __abs__.

    def randomise(self):
        """Initialise the lattice with random spins.

        Components of each spin are between -1 and 1: -1 <= si <= 1, and all
        spins are normalised to 1.

        """
        self.array = 2 * np.random.random((*self.n, 3)) - 1
        self.normalise()

    def plot(self):
        '''
        Plots the state of the Lattice which contains the atoms and shows
        a visual description and representation of their spins.
        This method returns a set of two plot containing the particle
        distribution and the topography
        '''
        # Defining our subplots
        fig, (arrow,topography)= plt.subplots(ncols=2,figsize=(12, 5), gridspec_kw={"hspace":5})
        # Grid of points
        xs = np.linspace(0, self.n[0]-1, self.n[0])
        ys = np.linspace(0, self.n[1]-1, self.n[1])
        x, y = np.meshgrid(xs, ys)
        # Write our rotational vector field (u,v,w)
        u = self.array[...,0]
        v = self.array[...,1]
        w = self.array[...,2]
        # Calling the atom plot method using a quiver technique to plot the particles
        atoms = arrow.quiver(x, y, u, v, w, pivot='middle',cmap="RdYlBu_r")
        # Calling the mountain plot using a contour filled to show the topography
        mountains = topography.contourf(w, levels=np.linspace(-1, 1.5, 11), cmap="RdYlBu_r")
        # Setting the Height bars
        direction = fig.colorbar(atoms, cmap="RdYlBu_r",ax=arrow,orientation='vertical')
        height = fig.colorbar(mountains,ax=topography,orientation='vertical')
        height.set_label('The height')
        direction.set_label('The Z Coordinate')
        