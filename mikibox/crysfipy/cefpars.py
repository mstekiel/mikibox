import numbers

class CEFpars:
    """Class representing set of crystal field parameters.

    It simplifies the creation of the CF parameter sets considering symmetry
    of the environment.
    Other modules expect that CF parameters are in *meV* (SI units). 
    But if you just want to diagonalize Hamiltonian, it is possible to use *K* (and results will be in *K*).

    Initialization can be done with named arguments or without. 
    If arguments are not named, symmetry is considered from the first string argument.
    Stevens parameters are considered differently for different symmetries in following order:

    | cubic: B40, B60
    | hexagonal: B20, B40, B44, B66
    | tetragonal: B20, B40, B44, B60, B64
    | orthorombic: B20, B22, B40, B42, B44, B60, B62, B64, B66
    

    Attributes:
        BXY (float, optional): Attribute corresponding to :math:`B_X^Y` Stevens Parameters.
            See Hutchings.  If at least one CF parameter is specified as a named argument, 
            non-named numerical parameters are ignored.
        sym (str, optional): Symmetry of the crystal field
            | c - cubic
            | h - hexagonal
            | t - tetragonal
            | o - orthorombic (default)
           
    Examples:
        Create set of CF parameters by named parameters:

        >>> print(cfpars(sym = "c", B40 = 10))
        Set of CF parameters for cubic symmetry:
        B40 = 10.0000
        B60 = 0.0000
        B44 = 50.0000
        B64 = 0.0000

        Use of non-named parameters:

        >>> print(cfpars("c", 10, 1))
        Set of CF parameters for cubic symmetry:
        B40 = 10.0000
        B60 = 1.0000
        B44 = 50.0000
        B64 = -21.0000

    """
    
    pars = {
    "c": ["cubic", ["B40", "B60", "B44", "B64"]],            
    "h": ["hexagonal", ["B20", "B40", "B44", "B66"]],              
    "t": ["tetragonal", ["B20", "B40", "B44", "B60", "B64"]],            
    "o": ["orthorombic", ["B20", "B22", "B40", "B42", "B44", "B60", "B62", "B64", "B66"]],  
    }
    
    def __init__(self, *args, **kwargs):
        # This is a poorly documented mess. Try to clean up.
        
        skipNonnamed = False
        self.sym = ""  #orthorombic symetry (=no constrains)
        #clear all params
        for name in self.pars["o"][1]: self._asignParameter(name, 0)
        #check named args
        for name, value in kwargs.items():
            #check symetry constrain
            if name == "sym":
                self.sym = self._asignSymmetry(value)    
            if name[0] == "B":  #it is a crystal field
                skipNonnamed = True
                self._asignParameter(name, value)
        #check non named args
        if self.sym == "":
            for value in args:
                if isinstance(value, str):
                    self.sym = self._asignSymmetry(value)
                    break  #just consider first string in list
        if self.sym == "": self.sym = "o" #default
        if not skipNonnamed:
            #read
            i = 0
            for value in args:
                if isinstance(value, numbers.Real): # this is the only place where `numbers` library is used. TODO to remove it
                    if i >= len(self.pars[self.sym][1]): break
                    self._asignParameter(self.pars[self.sym][1][i], value)
                    i+=1
        #do symmetry magic
        if self.sym == "c":
            self.B44 = 5 * self.B40;
            self.B64 = -21 * self.B60;
            
    def __str__(self):
        ret = "Set of CF parameters for %s symmetry:\n" % (self.pars[self.sym][0])
        for name in self.pars[self.sym][1]: ret += self._printParameter(name) + "\n"
        return ret
    
            
    def _asignSymmetry(self, inp):
        if inp[0] == "t" or  inp[0] == "h" or  inp[0] == "c":
            return inp[0]
        return ""
                  
    def _asignParameter(self, name, value):
        if name == "B20": self.B20 = value
        if name == "B22": self.B22 = value
        if name == "B40": self.B40 = value
        if name == "B42": self.B42 = value
        if name == "B44": self.B44 = value
        if name == "B60": self.B60 = value
        if name == "B62": self.B62 = value
        if name == "B64": self.B64 = value
        if name == "B66": self.B66 = value
        
    def _printParameter(self, name):
        if name == "B20": return "%s = %.4f" % (name, self.B20)
        if name == "B22": return "%s = %.4f" % (name, self.B22)
        if name == "B40": return "%s = %.4f" % (name, self.B40)
        if name == "B42": return "%s = %.4f" % (name, self.B42)
        if name == "B44": return "%s = %.4f" % (name, self.B44)
        if name == "B60": return "%s = %.4f" % (name, self.B60)
        if name == "B62": return "%s = %.4f" % (name, self.B62)
        if name == "B64": return "%s = %.4f" % (name, self.B64)
        if name == "B66": return "%s = %.4f" % (name, self.B66)