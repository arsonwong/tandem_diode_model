import numpy as np
from matplotlib import pyplot as plt

VT = 0.02568

class Diode:
    def __init__(self,I0=1e-15,n=1,V_shift=0): #V_shift is to shift the starting voltage, e.g. to define breakdown
        self.I0 = I0 #A
        self.n = n
        self.IL = 0
        self.V_shift = V_shift
    def __str__(self):
        return "I0 = " + str(self.I0) + " A, n = " + str(self.n)
    def solve_I(self, V, VT = VT):
        return self.I0*np.exp((V-self.V_shift)/(self.n*VT))

class SubCell:
    def __init__(self,diodes=[Diode()],rev_diodes=[],IL = 0, shunt_cond=0,name="subcell"):
        self.N_diodes = len(diodes)
        self.diodes = diodes
        self.rev_diodes = rev_diodes
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
        for diode in self.rev_diodes:
            I += diode.solve_I(-V,VT=VT) 
        if tabulate:
            V_ = np.copy(V)
            I_ = np.copy(I)   
            if V_[-1]>V_[0]:
                V_ = V_[::-1]
                I_ = I_[::-1]
            self.IV_table = np.vstack((V_,I_))  
        return I
    def plot(self):
        plt.plot(self.IV_table[0,:],self.IV_table[1,:])

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
        for i, cell in enumerate(self.subcells):
            if cell.IV_table is None:
                cell.solve_I(V, VT = VT, tabulate=True)
            if i==0:
                self.I_range = [np.min(cell.IV_table[1,:]),np.max(cell.IV_table[1,:])]
            else:
                self.I_range[0] = np.max([self.I_range[0],np.min(cell.IV_table[1,:])])
                self.I_range[1] = np.min([self.I_range[1],np.max(cell.IV_table[1,:])])
        if self.I_range[0] < self.I_range[1]: #tabulate the two extreme points
            self.solve_V(np.linspace(self.I_range[0],self.I_range[1],1000))
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
    def plot(self):
        for cell in self.subcells:
            cell.plot()
        plt.plot(self.IV_table[0,:],self.IV_table[1,:])
        plt.show()
    
def create_subcell(Isc, Voc, FF, n1, n2, n_rev, I0_rev=1e-15, breakdown_V=-10, VT=VT, name="subcell"):
    max_power = Isc*Voc*FF
    cell = SubCell(diodes=[Diode(n=n1),Diode(n=n2)],rev_diodes=[Diode(n=n_rev,V_shift=-breakdown_V)],IL = Isc, shunt_cond=0,name=name)
    lower_V = -n_rev*VT*np.log(Isc/I0_rev)+breakdown_V
    V = np.hstack((np.linspace(lower_V,breakdown_V,100), np.linspace(breakdown_V*0.99,breakdown_V*0.01,100), np.linspace(0, Voc+2*VT, 200)))
    # rough guess to start
    cell.diodes[0].I0 = Isc/np.exp(Voc/(n1*VT))
    cell.diodes[1].I0 = 0
    for i in range(100):
        I = cell.solve_I(V, tabulate=True)
        power = V*I
        Voc_ = np.interp(0.0, cell.IV_table[1,:], cell.IV_table[0,:])
        mpp = np.argmax(power)
        Vmp = V[mpp]
        max_power_ = power[mpp]
        denom = cell.diodes[0].solve_I(Voc_)/(n1*VT) + cell.diodes[1].solve_I(Voc_)/(n2*VT)
        dVoc_dI01 = -np.exp((Voc_-Vmp)/(n1*VT))/denom
        dVoc_dI02 = -np.exp((Voc_-Vmp)/(n2*VT))/denom
        dpower_dI01 = -Vmp
        dpower_dI02 = -Vmp
        M = np.array([[dVoc_dI01, dVoc_dI02],[dpower_dI01, dpower_dI02]])
        Y = np.array([Voc - Voc_, max_power-max_power_])
        x = np.linalg.solve(M, Y)
        cell.diodes[0].I0 += x[0]/np.exp(Vmp/(n1*VT))
        cell.diodes[1].I0 += x[1]/np.exp(Vmp/(n2*VT))
        if cell.diodes[0].I0 < 0:
            cell.diodes[0].I0 = 0
            cell.diodes[1].I0 = Isc/np.exp(Voc/(n2*VT))
            break
        if cell.diodes[1].I0 < 0:
            cell.diodes[1].I0 = 0
            cell.diodes[0].I0 = Isc/np.exp(Voc/(n1*VT))
            break
        if np.max(np.abs(Y))<1e-10:
            break
    return cell


    


