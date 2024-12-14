import numpy as np
from tandem_cells.diodes import Stack, SubCell, Diode

Vs = np.arange(0, 0.7 + 0.1, 0.1)

diode = Diode()
I = diode.solve_I(Vs)
print(I)

subcell = SubCell([Diode()]*2)
subcell.solve_I(np.arange(0, 0.7 + 0.1, 0.1),tabulate=True)
s = Stack([subcell]*8)

print(s)

s.tabulate_subcells(Vs)
print(s.I_range)
print(s.solve_V(-1* 10.0 ** np.arange(-3, -15, -1)))

print(s.IV_table)

print(s.solve_V(-1e-4))

print(s.solve_I(3))