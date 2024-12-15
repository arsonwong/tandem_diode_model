import numpy as np
from tandem_cells.diodes import Stack, SubCell, Diode, create_subcell

top_cell = create_subcell(21e-3, 1.3, 0.78, 1, 2, 1)
bottom_cell = create_subcell(20e-3, 0.7, 0.78, 1, 2, 1)
tandem_cell = Stack([top_cell,bottom_cell])
tandem_cell.tabulate_subcells(V=None)
V = np.linspace(0,2.2,100)
I = tandem_cell.solve_I(V)
print(I)