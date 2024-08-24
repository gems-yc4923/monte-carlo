'''
This is a module that calculates the total energy of a system containing a 2D Lattice of Spins.
It adds the zeeman energy, anisotropy, exchange and DMI energies into a total energy.

This Method accepts a system class of instance Spins.
It returns the sum of the total Energies of the system.

Example usage:
    system = mcsim.System(s=s, B=(0, 0, 0.1), K=0.01, u=(0, 0, 1), J=0.5, D=0.5)
    s is an instance of class Spins
    B is the external magnetic field vector
    K is the anisotropy constant
    u is the anisotropy vector
    J is the exchange energy constant
    D is the DMI constant


'''

import numpy as np

def normalise(v):
    '''
    This is a sub function that normalizes the input vectors

    Parameters
    ------------
    v: the vector input that we desire to normalize

    Returns
    ------------
    The vector v normalized
    '''
    #this is done in the traditional way. for some reason, linalg took much more time
    return v/(np.sqrt(sum(j**2 for j in v)))

class System:
    """System object with the spin configuration and necessary parameters.

    Parameters
    ----------
    s: mcsim.Spins

        Two-dimensional spin field.

    B: Iterable

        External magnetic field, length 3.

    K: numbers.Real

        Uniaxial anisotropy constant.

    u: Iterable(float)

        Uniaxial anisotropy axis, length 3. If ``u`` is not normalised to 1, it
        will be normalised before the calculation of uniaxial anisotropy energy.

    J: numbers.Real

        Exchange energy constant.

    D: numbers.Real

        Dzyaloshinskii-Moriya energy constant.

    """

    def __init__(self, s, B, K, u, J, D):
        '''
        Init function initializes the user inputs into the class created
        '''
        self.s = s
        self.J = J
        self.D = D
        self.B = B
        self.K = K
        self.u = u

    def energy(self):
        """Total energy of the system.

        The total energy of the system is computed as the sum of all individual
        energy terms.

        Returns
        -------
        float

            Total energy of the system.

        """
        return self.zeeman() + self.anisotropy() + self.exchange() + self.dmi()

    def zeeman(self):
        '''
        Calculate the sum of zeeman energies across all atoms
        '''
        #creating an empty list that will contain all the zeeman energies
        # of each atom
        sums = []
        for i in self.s.array:
            for j in i:
                #appending the zeeman energy of the selected atom
                sums.append(np.dot(j,self.B)*-1)
        return np.sum(sums)

    def anisotropy(self):
        '''
        Return the total uniaxial anisotropy energy of the system
        '''
        #creating an empty list that will contain all the anisotropy energies
        # of each atom
        result = []
        for i in self.s.array:
            for j in i:
                result.append(self.K*-1*(np.dot(j,
                                                normalise(self.u)))**2)
        return np.sum(result)

    def exchange(self):
        '''
        Return the total exchange energy between the spins
        '''
        #creating an empty list that will contain all the exchange energies
        #betwen atoms, either vertical exchange and horizontal exchange
        horizontal = []
        vertical = []
        for i in range(0,self.s.array.shape[0]):
            for j in range(0,self.s.array.shape[1]-1):
                horizontal.append(np.dot(self.s.array[i][j],
                                         self.s.array[i][j+1]))
        for j in range(0,self.s.array.shape[1]):
            for i in range(0,self.s.array.shape[0]-1):
                vertical.append(np.dot(self.s.array[i][j],
                                       self.s.array[i+1][j]))

        return -self.J*(sum(horizontal)+sum(vertical))

    def dmi(self):
        '''
        Return the total DMI energy between the spins
        ''' 
        #creating an empty list that will contain all the DMI energies
        #betwen atoms, either vertical exchange and horizontal exchange
        horizontal = []
        vertical = []
        #starting with a vertical/row iteration
        for i in range(0,self.s.array.shape[0]):
            for j in range(0,self.s.array.shape[1]-1):
                horizontal.append(np.dot(np.array([1,0,0]),
                                         (np.cross(self.s.array[i][j],
                                                   self.s.array[i][j+1]))))
        #ending with a horizontal/column iteration
        for j in range(0,self.s.array.shape[1]):
            for i in range(0,self.s.array.shape[0]-1):
                vertical.append(np.dot(np.array([0,-1,0]),
                                       (np.cross(self.s.array[i][j],
                                                 self.s.array[i+1][j]))))

        return self.D*(sum(horizontal)+sum(vertical))
        
