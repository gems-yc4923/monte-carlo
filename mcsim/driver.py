'''
This is a module that runs the Monte Carlo Simulation that will serve to minimise the 
total energy of the system until it reaches a minimum, depending on the system's initial state.

This Method accepts a system class of instance System and does not return anything, as the
changes are automatically done to the system.

Example usage:
    driver = mcsim.Driver() # Initialise the driver class
    driver.drive(system, n=10_000) #Run the Simulation with n=10000 simulations

'''

import numpy as np
def random_spin(s0, alpha=0.1):
    """Generate a new random spin based on the original one.

    Parameters
    ----------
    s0: np.ndarray

        The original spin that needs to be changed.

    alpha: float

        Larger alpha, larger the modification of the spin. Defaults to 0.1.

    Returns
    -------
    np.ndarray

        New updated spin, normalised to 1.

    """
    delta_s = (2 * np.random.random(3) - 1) * alpha
    s1 = s0 + delta_s
    return s1 / np.linalg.norm(s1)

class Driver:
    """Driver class.

    Driver class does not take any input parameters at initialisation.

    """

    def __init__(self):
        pass

    def drive(self, system, n, alpha=0.1):
        """Initializes the Monte Carlo Simulation

        Parameters
        ----------
        system: System Class

            The System that is automatically passed.
        
        n: integer

            The Number of total iterations. The Bigger the number, the longer the simulation.

        alpha: float

            Larger alpha, larger the modification of the spin. Defaults to 0.1.

        """
        for _ in range(n):
            #taking the number of rows and columns
            ij = (system.s.array.shape[0],system.s.array.shape[1])
            #outputing a random column and row number
            i = np.abs(int(((np.random.random()*2)-1)*ij[0]))
            j = np.abs(int(((np.random.random()*2)-1)*ij[1]))
            # Spin number {i},{j} has value {sij}
            #calculating the current energy
            e0 = system.energy()
            #creating a backup of the current system
            backup = np.copy(system.s.array)
            #changing the random spin in our current system
            system.s.array [i][j] = random_spin(system.s.array[i,j,:],alpha)
            e1 = system.energy()
            # Now we compare the difference between the old and the new
            # If changes are rejected, revert to the old system
            # stored in the backup
            if e1>e0:
                system.s.array = backup
