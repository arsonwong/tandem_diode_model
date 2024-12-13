class Diode:
    def __init__(self,I0=1e-15,n=1):
        self.I0 = I0 #A
        self.n = n
    def __str__(self):
        return "I0 = " + str(self.I0) + " A, n = " + str(self.n)

class SubCell:
    def __init__(self,diodes=[Diode()],shunt_cond=0,name="subcell"):
        self.N_diodes = len(diodes)
        self.diodes = diodes
        for i, diode in enumerate(diodes):
            if not isinstance(diode,Diode):
                self.diodes[i] = Diode()
        self.shunt_cond = shunt_cond #1/ohm
        self.name = name
    def __str__(self):
        word = ""
        for i, diode in enumerate(self.diodes):
            word += "Diode " + str(i) + ": " + str(diode) + "\n"
        word += "Shunt cond: " + str(self.shunt_cond) + " ohm-1\n"
        return word

class Stack:
    def __init__(self, subcells=[SubCell()], Rs=0):
        self.N = len(subcells)
        self.subcells = subcells
        for i, cell in enumerate(subcells):
            if not isinstance(cell,SubCell):
                self.subcells[i] = SubCell()
        self.Rs = Rs #ohm
    def __str__(self):
        word = "Stack Rs = " + str(self.Rs) + " ohm\n"
        for i, subcell in enumerate(self.subcells):
            word += "---------------------\n"
            word += "Subcell " + str(i) + " (" + subcell.name + "): \n"
            word += str(subcell)
        return word