import numpy as np
from tandem_cells.diodes import Stack, SubCell, Diode

subcell = SubCell([Diode()]*2)
d = Stack([subcell]*8)

print(d)