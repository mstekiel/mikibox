For systems with J>=5/2 the diagonalization of the Hamiltonian takes long time.

Ideas how to improve:
1. Calculate only eigenvalues -> thats actually super fast!!!
2. Start by reducing columns and rows -> Seem like sympy is smart enough to do that on its own

Ideas how to analyse
1. If the system is overdefined one can assume some parameters are small, and substitute them in equations to 0. This will produce a system with reduced number of parameters which will act as a initial set of parameters that can be later fit
2. Fin out about the physical boundaries on the parameters, like I think B_20>0