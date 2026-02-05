import numpy as np

"""
A set of classes for simulating the behaviour of loop extruding factors (LEFs) on a one-dimensional lattice
Originally developed by the Open2C community; built upon the code from `polychrom` examples (https://github.com/open2c/polychrom/blob/master/examples/loopExtrusion/extrusion_1D_newCode.ipynb)
"""

########################
# CLASSES
########################


class head(object):
    def __init__(self, pos, attrs={"stalled": False, "CTCF": False}):
        """
        A LEF head is described by its positon on the lattice and its state, i.e., whether it is free, stalled, or captured by a CTCF
        """
        self.pos = pos
        self.attrs = dict(attrs)

########################


class LEF(object):

    """
    The LEF class provides fast access to its position and state
    """
    
    def __init__(self, head1, head2):
        self.left = head1
        self.right = head2
   
    def any(self, attr):
        return self.left.attrs[attr] or self.right.attrs[attr]
    
    def all(self, attr):
        return self.left.attrs[attr] and self.right.attrs[attr]
    
    def __getitem__(self, item):
        if item == -1:
            return self.left
        elif item == 1:
            return self.right 
        else:
            raise ValueError()


########################
# FUNCTIONS
########################
        

def unload_prob(LEF:object, args:dict) -> float:
    
    """
    Assign a different unload probability for LEFs based on their state or at the boundary of the system
    """

    # unload a LEF when it is reaches either end of the system
    if LEF.left.pos == 1 or LEF.right.pos == args["lattice_length"] - 2:
        return 1
    # preferential unloading in the buffer zone, inactive by default
    if LEF.left.pos in args["buffer_zone"] or LEF.right.pos in args["buffer_zone"]:
        return args["LEF_unload_prob"]
    # twice the probability of LEF unloading when stalled
    if LEF.any("stalled"):
        return args["LEF_stalled_unload_prob"]
    # otherwise return the predefined probability
    return args["LEF_unload_prob"]

########################


def load(LEFs:list, occupied:np.ndarray, args:dict) -> None:
    """
    Loads a LEF to a random unoccupied position on the lattice
    """
    while True:
        a = np.random.randint(args["lattice_length"])
        if (occupied[a] == 0) and (occupied[a+1] == 0):
            occupied[a] = 1
            occupied[a+1] = 1
            LEFs.append(LEF(head(a), head(a+1)))
            break

########################


def capture(LEF:object, args:dict) -> object:
    """
    Capture a LEF head upon encountering a CTCF based on a pre-defined probability
    """    
    for side in [1, -1]:
        # get probability of capture or otherwise it is 0
        if np.random.uniform() < args["CTCF_capture"][side].get(LEF[side].pos, 0):
            LEF[side].attrs["stalled"] = False
            LEF[side].attrs["CTCF"] = True  # captured a LEF head at CTCF
    return LEF

########################


def release(LEF:object, args: dict) -> object:
    """
    Release a captured LEF head from a CTCF based on a pre-defined probabaility
    """
    if not LEF.any("CTCF"):
        return LEF  # if the LEF is not captured, nothing is done
        
    # release the LEF head which is captured
    for side in [-1, 1]:
        if (np.random.uniform() < args["CTCF_release"][side].get(LEF[side].pos, 0)) and (LEF[side].attrs["CTCF"]):
            LEF[side].attrs["CTCF"] = False
    return LEF

########################


def translocate(LEFs:list, occupied:np.ndarray, args:dict):
    """
    Combines the unloading, loading, capture, and release of LEFs on the lattice and then performs translocation of LEFs to mimick loop extrusion
    """

    # # NOTE naive approach to control dynamicity, load a LEF if there are none in the system, unload a few if there are too many
    # elif len(LEFs) > 5 * args["initial_num_LEFs"]:
    #     for _ in range(np.random.randint(1, len(LEFs) // 5)):
    #         i = np.random.randint(len(LEFs))
    #         occupied[LEFs[i].left.pos] = 0
    #         occupied[LEFs[i].right.pos] = 0
    #         del LEFs[i]

    # unloading LEFs
    num_LEFs_current = len(LEFs)
    idxs_to_unload = []
    for i in range(num_LEFs_current):
        if np.random.uniform() < unload_prob(LEFs[i], args):
            idxs_to_unload.append(i)
    idxs_to_unload.reverse()
    for idx in idxs_to_unload:
        occupied[LEFs[idx].left.pos] = 0
        occupied[LEFs[idx].right.pos] = 0
        del LEFs[idx]
    
    # loading LEFs
    if args["constant_LEF_occupancy"] and len(idxs_to_unload) > 0:      # load LEFs to maintain a constant number of LEFs if it is kept constant
        for _ in range(len(idxs_to_unload)):
            load(LEFs, occupied, args)
    # elif not args["constant_LEF_occupancy"]:
        # if len(LEFs) == 0:  # load a LEF if there are none in the system
        #     load(LEFs, occupied, args)
        # # for i in range(num_LEFs_current):
        #     if np.random.uniform() < args["LEF_load_prob"]:      # load a LEF on the lattice based on some probability if number of LEFs is not constant
        #         load(LEFs, occupied, args)

    elif not args["constant_LEF_occupancy"] and np.random.uniform() < args["LEF_load_prob"]:      # load a LEF on the lattice based on some probability if number of LEFs is not constant
        load(LEFs, occupied, args)
    
    # capture and release LEFs at CTCF sites 
    for i in range(len(LEFs)):
        LEFs[i] = capture(LEFs[i], args)
        LEFs[i] = release(LEFs[i], args)
    
    # translocate the LEFs
    for i in range(len(LEFs)):
        for head in [-1, 1]:
            # taking only LEFs that are not captured at CTCFs
            if not LEFs[i][head].attrs["CTCF"]:
                # considering only colliding LEFs
                if occupied[LEFs[i][head].pos + head] != 0:
                    if LEFs[i][head].pos + head not in list(range(0, 10)) + list(range(args["lattice_length"] - 10, args["lattice_length"])) and np.random.uniform() < args["LEF_traversal_prob"]:
                        # the LEF has to traverse everything until the next unoccupied position on the lattice
                        occupied[LEFs[i][head].pos] = 0
                        start, end = sorted([LEFs[i][head].pos + head, LEFs[i][head].pos + (head * 10)])
                        # move the LEF to its new position and mark it as unstalled
                        LEFs[i][head].attrs["stalled"] = False
                        LEFs[i][head].pos += np.argmax(occupied[start:end] == 0) * head
                        occupied[LEFs[i][head].pos] = 1
                    else:
                        # mark LEF head as stalled
                        LEFs[i][head].attrs["stalled"] = True
                        
                else:
                    # otherwise translocate the LEF head with a probability to mimic velocity
                    LEFs[i][head].attrs["stalled"] = False
                    if np.random.uniform() < args["LEF_step_prob"]:
                        occupied[LEFs[i][head].pos] = 0
                        occupied[LEFs[i][head].pos + head] = 1
                        LEFs[i][head].pos += head

########################


def state(LEFs, args):
    """
    Color LEFs by their state given a list of LEFs
    """
    def state(attrs):
        if attrs["stalled"] and attrs["CTCF"]:
            raise ValueError("LEFs cannot be both captured and stalled simultaneously - problem in simulations")
        if attrs["stalled"]:
            return 2
        if attrs["CTCF"]:
            return 3
        return 1
    ar = np.zeros(args["lattice_length"])
    for l in LEFs:
        ar[l.left.pos] = state(l.left.attrs)
        ar[l.right.pos] = state(l.right.attrs)  
    return ar

########################


def root_LEFs(LEFs, args):
   
    """
    Retrieve only the positions of root LEFs, i.e., those that have no nested LEFs in between their two heads on the lattice
    """

    # create and fill a lattice based on LEF positions to indicate loops
    occupied = np.zeros(args["lattice_length"], dtype=int)
    occupied[[LEFs[i][0] for i in range(len(LEFs))]] = -1
    occupied[[LEFs[i][1] for i in range(len(LEFs))]] = 1
    
    # use a stack to check for root loops
    stack = []
    root_LEFs = []
    for i in range(len(occupied)):
        if occupied[i] != 0:
            if occupied[i] == -1:
                stack.append(i)
            else:
                lo_cohesin = stack.pop()
                if len(stack) == 0:
                    root_LEFs.append((lo_cohesin, i))
    if stack:
        raise Exception("LEFs are not nested properly; cannot find roots")
    
    return root_LEFs

########################


def _LEF_occupancy(LEFs, args):
    """Get the occupancy of LEFs per 100 sites at each time step throughout a simulation"""
    return [len(arr) * (100 / args["lattice_length"]) for arr in LEFs]

########################


def mean_LEF_occupancy(LEFs, args):
    """Compute the mean occupancy of LEFs per 100 sites during a simulation"""
    return np.mean(_LEF_occupancy(LEFs, args))

########################


##### For simulations on a polymer

class extruder(object):

    """
    A bond updater for loop extrusion during the molecular dynamics simulations of chromatin.
    Uses a lattice-based simulation trajectory of loop extrusion to move harmonic bonds between monomers that are held together by LEF heads over time.
    Originally developed by the Open2C community; borrowed from `polychrom` examples (https://github.com/open2c/polychrom/blob/master/examples/loopExtrusion/extrusion_3D.ipynb)
    """

    def __init__(self, LEFpositions):

        """
        :param smcTransObject: smc translocator object to work with
        """
        self.LEFpositions = LEFpositions
        self.curtime  = 0
        self.allBonds = []


    def setParams(self, activeParamDict, inactiveParamDict):

        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce, blocks=100):

        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """

        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce

        # precalculating bonds upto a specific number of blocks
        if type(self.LEFpositions) == np.ndarray:       # if constant LEF occupancy
            allBonds = []
            loaded_positions  = self.LEFpositions[self.curtime:self.curtime+blocks]
            allBonds = [[(int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1])) 
                            for j in range(loaded_positions.shape[1])] for i in range(blocks)]
        else:
            allBonds = self.LEFpositions[self.curtime:self.curtime+blocks]      # if variable LEF occupancy

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset) # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        
        self.curtime += blocks 
        
        return self.curBonds,[]


    def step(self, context, verbose=False):

        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """

        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; restart the simulation and run setup again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        return self.curBonds, pastBonds