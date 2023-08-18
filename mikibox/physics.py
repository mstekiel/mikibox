import numpy as np

def bose_occupation(energies: np.ndarray, temperature: float) -> np.ndarray:
    '''
    Calculate occupation number math:`n(E,T)` within Bose statstics.
    math:`n(E,T) = 1/(exp(E/k_B T)-1)`

    energies:
        Array of energies in math:`meV`.
    temperature:
        Temperature in math:`K`.
    '''
    meV2K = 11.60452
    return 1/(np.exp(energies*meV2K/temperature)-1)

# For fast testing
if __name__ == '__main__':
    energies = np.linspace(5, 13, 17)
    T1 = 80
    T2 = 15

    print(bose_occupation(energies, 15))
    print(bose_occupation(energies, T1) / bose_occupation(energies, T2))