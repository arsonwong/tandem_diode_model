import numpy as np

VT = 0.02568

class Diode:
    def __init__(self,I0=1e-15,n=1):
        self.I0 = I0 #A
        self.n = n
        self.IL = 0
    def __str__(self):
        return "I0 = " + str(self.I0) + " A, n = " + str(self.n)
    def solve_I(self, V, VT = VT):
        return self.I0*np.exp(V/(self.n*VT))

class SubCell:
    def __init__(self,diodes=[Diode()],IL = 0, shunt_cond=0,name="subcell"):
        self.N_diodes = len(diodes)
        self.diodes = diodes
        for i, diode in enumerate(diodes):
            if not isinstance(diode,Diode):
                self.diodes[i] = Diode()
        self.shunt_cond = shunt_cond #1/ohm
        self.name = name
        self.IL = IL
        self.IV_table = None
    def __str__(self):
        word = ""
        for i, diode in enumerate(self.diodes):
            word += "Diode " + str(i) + ": " + str(diode) + "\n"
        word += "Shunt cond: " + str(self.shunt_cond) + " ohm-1\n"
        return word
    def solve_I(self, V, IL = None, VT = VT, tabulate=False):
        if IL is not None:
            self.IL = IL
        I = self.IL - V*self.shunt_cond
        for diode in self.diodes:
            I -= diode.solve_I(V,VT=VT) 
        if tabulate:
            indices = np.argsort(I)
            V = V[indices]
            I = I[indices]     
            self.IV_table = np.vstack((V,I))  
        return I

class Stack:
    def __init__(self, subcells=[SubCell()], Rs=0):
        self.N = len(subcells)
        self.subcells = subcells
        for i, cell in enumerate(subcells):
            if not isinstance(cell,SubCell):
                self.subcells[i] = SubCell()
        self.Rs = Rs #ohm
        self.I_range = None
        self.IV_table = None
    def __str__(self):
        word = "Stack Rs = " + str(self.Rs) + " ohm\n"
        for i, subcell in enumerate(self.subcells):
            word += "---------------------\n"
            word += "Subcell " + str(i) + " (" + subcell.name + "): \n"
            word += str(subcell)
        return word
    def tabulate_subcells(self, V, VT = VT):
        for i, _ in enumerate(self.subcells):
            I = self.subcells[i].solve_I(V, VT = VT, tabulate=True)
            if i==0:
                self.I_range = [np.min(I),np.max(I)]
            else:
                self.I_range[0] = np.max([self.I_range[0],np.min(I)])
                self.I_range[1] = np.min([self.I_range[1],np.max(I)])
        if self.I_range[0] < self.I_range[1]: #tabulate the two extreme points
            self.solve_V(np.array(self.I_range))
    def solve_V(self,I):
        if not isinstance(I, np.ndarray):
            I = np.array([I])
        V = np.ones_like(I) * np.nan
        indices = np.where((I >= self.I_range[0]) & (I <= self.I_range[1]))[0]
        V[indices] = -I[indices]*self.Rs
        for i, cell in enumerate(self.subcells):
            V[indices] += np.interp(I[indices], cell.IV_table[1,:], cell.IV_table[0,:])
        subtable = np.vstack((V[indices],I[indices]))  
        if self.IV_table is None:
            self.IV_table = subtable
        else:
            self.IV_table = np.hstack((self.IV_table,subtable))
            indices = np.argsort(self.IV_table[0,:])
            self.IV_table = self.IV_table[:,indices]
        return V
    def solve_I(self,V,V_tolerance=1e-5):
        if not isinstance(V, np.ndarray):
            V = np.array([V])
        I = np.ones_like(V) * np.nan
        indices = np.where((V >= min(self.IV_table[0,:])) & (V <= max(self.IV_table[0,:])))[0]
        while True:
            I[indices] = np.interp(V[indices], self.IV_table[0,:], self.IV_table[1,:])
            Vnew = self.solve_V(I[indices])
            if np.max(np.abs(Vnew-V[indices])) < V_tolerance:
                break
        return I

    


