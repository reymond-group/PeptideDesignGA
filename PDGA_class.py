import os
import random
import subprocess as sub
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops


# pop_size, mut_rate, gen_gap, query, sim treshold

def cyclize(mol, cy):
    """it is connecting cyclizing the given molecule

    Arguments:
        mol {rdKit mol object} -- molecule to be cyclized
        cy {int} -- 1=yes, 0=no cyclazation

    Returns:
        mols {list of rdKit mol objects} -- possible cyclazation
    """
    count = 0

    # detects all the N terminals in mol
    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
            count += 1
            atom.SetProp('Nterm', 'True')
        else:
            atom.SetProp('Nterm', 'False')

    # detects all the C terminals in mol (it should be one)
    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
            atom.SetProp('Cterm', 'True')
        else:
            atom.SetProp('Cterm', 'False')

    # detects all the S terminals in mol

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:1]':
            atom.SetProp('Sact1', 'True')
        else:
            atom.SetProp('Sact1', 'False')

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:2]':
            atom.SetProp('Sact2', 'True')
        else:
            atom.SetProp('Sact2', 'False')

    for atom in mol.GetAtoms():
        if atom.GetSmarts() == '[S:3]':
            atom.SetProp('Sact3', 'True')
        else:
            atom.SetProp('Sact3', 'False')

    Nterm = []
    Cterm = []
    Sact1 = []
    Sact2 = []
    Sact3 = []

    # saves active Cysteins postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact1') == 'True':
            Sact1.append(atom.GetIdx())

    # saves active Cysteins 2 postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact2') == 'True':
            Sact2.append(atom.GetIdx())

    # saves active Cysteins 3 postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Sact3') == 'True':
            Sact3.append(atom.GetIdx())

    # creates the S-S bond (in the current version only two 'active' Cys, this codo picks two random anyway):
    while len(Sact1) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact1)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact1[x]
        b = Sact1[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact1', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact1', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact1.remove(a)
        Sact1.remove(b)

    while len(Sact2) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact2)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact2[x]
        b = Sact2[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact2', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact2', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact2.remove(a)
        Sact2.remove(b)

    while len(Sact3) >= 2:
        edmol = rdchem.EditableMol(mol)
        pos = list(range(len(Sact3)))
        x = np.random.choice(pos, 1)[0]
        pos.remove(x)
        y = np.random.choice(pos, 1)[0]
        a = Sact3[x]
        b = Sact3[y]
        edmol.AddBond(a, b, order=Chem.rdchem.BondType.SINGLE)
        mol = edmol.GetMol()
        mol.GetAtomWithIdx(a).SetProp('Sact3', 'False')
        mol.GetAtomWithIdx(b).SetProp('Sact3', 'False')
        mol.GetAtomWithIdx(a).SetAtomMapNum(0)
        mol.GetAtomWithIdx(b).SetAtomMapNum(0)
        Sact3.remove(a)
        Sact3.remove(b)

    # saves active C and N terminals postions:
    for atom in mol.GetAtoms():
        if atom.GetProp('Nterm') == 'True':
            Nterm.append(atom.GetIdx())
        if atom.GetProp('Cterm') == 'True':
            Cterm.append(atom.GetIdx())

    if cy == 1:
        edmol = rdchem.EditableMol(mol)

        # creates the amide bond
        edmol.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
        edmol.RemoveAtom(Cterm[0] + 1)

        mol = edmol.GetMol()

        # removes tags and lables form the atoms which reacted
        mol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
        mol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
        mol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)
        mol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)

    return mol


def attach_capping(mol1, mol2):
    """it is connecting all Nterminals with the desired capping

    Arguments:
        mol1 {rdKit mol object} -- first molecule to be connected
        mol2 {rdKit mol object} -- second molecule to be connected - chosen N-capping

    Returns:
        rdKit mol object -- mol1 updated (connected with mol2, one or more)
    """

    count = 0

    # detects all the N terminals in mol1
    for atom in mol1.GetAtoms():
        atom.SetProp('Cterm', 'False')
        if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
            count += 1
            atom.SetProp('Nterm', 'True')
        else:
            atom.SetProp('Nterm', 'False')

    # detects all the C terminals in mol2 (it should be one)
    for atom in mol2.GetAtoms():
        atom.SetProp('Nterm', 'False')
        if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
            atom.SetProp('Cterm', 'True')
        else:
            atom.SetProp('Cterm', 'False')

    # mol2 is addes to all the N terminal of mol1
    for i in range(count):
        combo = rdmolops.CombineMols(mol1, mol2)
        Nterm = []
        Cterm = []

        # saves in two different lists the index of the atoms which has to be connected
        for atom in combo.GetAtoms():
            if atom.GetProp('Nterm') == 'True':
                Nterm.append(atom.GetIdx())
            if atom.GetProp('Cterm') == 'True':
                Cterm.append(atom.GetIdx())

        # creates the amide bond
        edcombo = rdchem.EditableMol(combo)
        edcombo.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
        clippedMol = edcombo.GetMol()

        # removes tags and lables form the atoms which reacted
        clippedMol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
        clippedMol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
        clippedMol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)
        clippedMol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)
        # uptades the 'core' molecule
        mol1 = clippedMol

    return mol1


def calc_cbd(fp1, fp2):
    """Calculates manhattan distance (cbd) between two given fp

    Arguments:
        fp1 {string} -- mxfp
        fp2 {string} -- mxfp

    Returns:
        int -- cbd
    """

    FP2 = list(map(int, fp2.split(';')))
    FP1 = list(map(int, fp1.split(';')))
    cbd = np.sum(np.abs(np.array(FP2) - np.array(FP1)))

    return cbd


def remove_duplicates(gen):
    """Removes duplicates

    Arguments:
        gen {list} -- sequences

    Returns:
        list -- unique list of sequences
    """

    gen_u = []
    for seq in gen:
        if seq not in gen_u:
            gen_u.append(seq)
    return gen_u


def mating(parents):
    """splits the parents in half and join them giving a child

    Arguments:
        parents {list of strings} -- parents

    Returns:
        string -- child
    """

    parent1 = parents[0]
    parent2 = parents[1]
    half1 = parent1[:random.randint(int(round(len(parent1) / 2, 0)) - 1, int(round(len(parent1) / 2, 0)) + 1)]
    half2 = parent2[random.randint(int(round(len(parent2) / 2, 0)) - 1, int(round(len(parent2) / 2, 0)) + 1):]
    child = half1 + half2

    for i in ['-------','------','-----','----','---','--']:
        if i in child:
            child = child.replace(i, '-')

    if 'X' in child[1:]:
        child_tmp = child[:1] + child[1:].replace('X','')
        child = child_tmp

    if '&' in child[1:]:
        child_tmp = child[:1] + child[1:].replace('&','')
        child = child_tmp
    
    if '+' in child[:-1]:    
        child_tmp = child[:-1].replace('+','') + child[-1:]
        child = child_tmp

    return child


def swapcy(seq):
    """insertion of two ativated cys at head to tail position

    Arguments:
        seq {string} -- peptide seq

    Returns:
        string -- S-S cyclized peptide seq
    """

    act_cys = 'Ä'
    if 'Ä' in seq:
        act_cys = 'Ö'
        if 'Ö' in seq:
            act_cys = 'Ü'
            if 'Ü' in seq:
                return seq

    new_seq = act_cys + seq[1:] + act_cys

    return new_seq


def break_SS(seq):
    """inactivation of all cys

    Arguments:
        seq {string} -- peptide seq

    Returns:
        string -- S-S cyclized peptide seq
    """

    act_cys = 'Ü'
    if 'Ü' not in seq:
        act_cys = 'Ö'
        if 'Ö' not in seq:
            act_cys = 'Ä'
            if 'Ä' not in seq:
                return seq

    # seq.replace('Ä', 'C')
    # seq.replace('Ü', 'C')
    # seq.replace('Ö' ,'C')
    seq.replace(act_cys, '')

    return seq


def set_seed(seed):
    """set seed for random

    Arguments:
        seed {int} -- sed for random
    """

    random.seed(int(seed))
    np.random.seed(int(seed))


class PDGA:
    interprete_dict = {'Arg': 'R', 'His': 'H', 'Lys': 'K', 'Asp': 'D', 'Glu': 'E', 'Ser': 'S', 'Thr': 'T', 'Asn': 'N',
                       'Gln': 'Q', 'Cys': 'C', 'Sec': 'U', 'Gly': 'G', 'Pro': 'P', 'Ala': 'A', 'Ile': 'I', 'Leu': 'L',
                       'Met': 'M', 'Phe': 'F', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V', 'Dap': '1', 'Dab': '2',
                       'BOrn': '3', 'BLys': '4', 'Hyp': 'Z', 'Orn': 'O', 'bAla': '!', 'Gaba': '?', 'dDap': '5',
                       'dDab': '6',
                       'dBOrn': '7', 'dBLys': '8', 'dArg': 'r', 'dHis': 'h', 'dLys': 'k', 'dAsp': 'd', 'dGlu': 'e',
                       'dSer': 's',
                       'dThr': 't', 'dAsn': 'n', 'dGln': 'q', 'dCys': 'c', 'dSec': 'u', 'dGly': 'g', 'dPro': 'p',
                       'dAla': 'a',
                       'dIle': 'i', 'dLeu': 'l', 'dMet': 'm', 'dPhe': 'f', 'dTrp': 'w', 'dTyr': 'y', 'dVal': 'v',
                       'dHyp': 'z', 'dOrn': 'o', 'a5a': '=', 'a6a': '%', 'a7a': '$', 'a8a': '@', 'a9a': '#',
                       'Cys1': 'Ä', 'Cys2': 'Ö', 'Cys3': 'Ü', 'dCys1': 'ä', 'dCys2': 'ö', 'dCys3': 'ü',
                       'Ac': '&', 'NH2': '+', 'met': '-', 'cy': 'X'}

    interprete_rev_dict = {v: k for k, v in interprete_dict.items()}

    # list of possible aminoacids
    AA = ['R', 'H', 'K', 'E', 'S', 'T', 'N', 'Q', 'G', 'P', 'A', 'V', 'I', 'L', 'F', 'Y', 'W', 'C', 'D', 'M', 'Z', 'O',
          '!', '?', '=', '%', '$', '@', '#']
    # list of possible branching units (1=Dap, 2=Dab, 3=Orn, 4=Lys)
    B = ['1', '2', '3', '4']
    # list of possible C-terminals
    CT = ['+']
    # list of possible N-capping
    NT = ['&']

    # variables for random generation of dendrimers
    AA4rndm = ['O', 'Z', 'R', 'H', 'K', 'E', 'S', 'T', 'N', 'Q', 'G', 'P', 'A', 'V', 'I', 'L', 'F', 'Y', 'W', 'C', 'D',
               'M',
               '!', '?', '=', '%', '$', '@', '#', '', '', '', '', '', '', '',
               '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
    B4rndm = ['1', '2', '3', '4', '']
    CTrndm = ['+', '']
    NTrndm = ['&', '']

    # B4rndm = [''] (only linear generation)
    max_aa_no = 5
    max_gen_no = 3

    # variables for SMILES generation
    B_SMILES = {'1': '[N:2]C(C[N:2])[C:1](O)=O', '2': '[N:2]C(CC[N:2])[C:1](O)=O',
                '3': '[N:2]C(CCC[N:2])[C:1](O)=O', '4': '[N:2]C(CCCC[N:2])[C:1](O)=O',
                '5': '[N:2]C(C[N:2])[C:1](O)=O', '6': '[N:2]C(CC[N:2])[C:1](O)=O',
                '7': '[N:2]C(CCC[N:2])[C:1](O)=O', '8': '[N:2]C(CCCC[N:2])[C:1](O)=O'}

    AA_SMILES = {'A': '[N:2]C(C)[C:1](O)=O', 'R': '[N:2]C(CCCNC(N)=N)[C:1](O)=O',
                 'N': '[N:2]C(CC(N)=O)[C:1](O)=O', 'D': '[N:2]C(CC(O)=O)[C:1](O)=O',
                 'C': '[N:2]C(CS)[C:1](O)=O', 'Q': '[N:2]C(CCC(N)=O)[C:1](O)=O',
                 'E': '[N:2]C(CCC(O)=O)[C:1](O)=O', 'G': '[N:2]C[C:1](O)=O',
                 'H': '[N:2]C(CC1=CNC=N1)[C:1](O)=O', 'I': '[N:2]C(C(C)CC)[C:1](O)=O',
                 'K': '[N:2]C(CCCCN)[C:1](O)=O', 'L': '[N:2]C(CC(C)C)[C:1](O)=O',
                 'M': '[N:2]C(CCSC)[C:1](O)=O', 'F': '[N:2]C(CC1=CC=CC=C1)[C:1](O)=O',
                 'P': 'C1CC[N:2]C1[C:1](O)=O', 'S': '[N:2]C(CO)[C:1](O)=O',
                 'T': '[N:2]C(C(O)C)[C:1](O)=O', 'W': '[N:2]C(CC1=CNC2=CC=CC=C12)[C:1](O)=O',
                 'Y': '[N:2]C(CC1=CC=C(C=C1)O)[C:1](O)=O', 'V': '[N:2]C(C(C)C)[C:1](O)=O',
                 'Ä': '[N:2]C(C[S:1])[C:1](O)=O', 'Ö': '[N:2]C(C[S:2])[C:1](O)=O',
                 'Ü': '[N:2]C(C[S:3])[C:1](O)=O',
                 'Z': 'C1C(O)C[N:2]C1[C:1](O)=O',
                 'O': '[N:2]C(CCCN)[C:1](O)=O',
                 'a': '[N:2]C(C)[C:1](O)=O', 'r': '[N:2]C(CCCNC(N)=N)[C:1](O)=O',
                 'n': '[N:2]C(CC(N)=O)[C:1](O)=O', 'd': '[N:2]C(CC(O)=O)[C:1](O)=O',
                 'c': '[N:2]C(CS)[C:1](O)=O', 'q': '[N:2]C(CCC(N)=O)[C:1](O)=O',
                 'e': '[N:2]C(CCC(O)=O)[C:1](O)=O', 'g': '[N:2]C[C:1](O)=O',
                 'h': '[N:2]C(CC1=CNC=N1)[C:1](O)=O', 'i': '[N:2]C(C(C)CC)[C:1](O)=O',
                 'k': '[N:2]C(CCCCN)[C:1](O)=O', 'l': '[N:2]C(CC(C)C)[C:1](O)=O',
                 'm': '[N:2]C(CCSC)[C:1](O)=O', 'f': '[N:2]C(CC1=CC=CC=C1)[C:1](O)=O',
                 'p': 'C1CC[N:2]C1[C:1](O)=O', 's': '[N:2]C(CO)[C:1](O)=O',
                 't': '[N:2]C(C(O)C)[C:1](O)=O', 'w': '[N:2]C(CC1=CNC2=CC=CC=C12)[C:1](O)=O',
                 'y': '[N:2]C(CC1=CC=C(C=C1)O)[C:1](O)=O', 'v': '[N:2]C(C(C)C)[C:1](O)=O',
                 'ä': '[N:2]C(C[S:1])[C:1](O)=O', 'ö': '[N:2]C(C[S:2])[C:1](O)=O',
                 'ü': '[N:2]C(C[S:3])[C:1](O)=O',
                 '!': '[N:2]CC[C:1](O)=O', '?': '[N:2]CCC[C:1](O)=O',
                 '=': '[N:2]CCCC[C:1](O)=O', '%': '[N:2]CCCCC[C:1](O)=O',
                 '$': '[N:2]CCCCCC[C:1](O)=O', '@': '[N:2]CCCCCCC[C:1](O)=O',
                 '#': '[N:2]CC[C:1](O)=O'}

    T_SMILES = {'+': '[N:2]'}

    C_SMILES = {'&': 'C[C:1](=O)'}

    # GA class var
    mut_n = 1
    b_insert_rate = 0.1
    selec_strategy = 'Elitist'
    rndm_newgen_fract = 10
    fitness = 'MXFP'

    # initiatization of class variables updated by the GA
    dist_dict_old = {}
    gen_n = 0
    found_identity = 0
    steady_min = 0
    timelimit_seconds = None
    cbd_av = None
    cbd_min = None
    dist_dict = None
    surv_dict = None
    time = 0
    min_dict = {}
    # used internally to recognize a methylated aa:
    metbond = False
    # can be set with exclude or allow methylation, 
    # it refers to the possibility of having methylation in the entire GA:
    methyl = False

    # debug
    verbose = False

    def __init__(self, pop_size, mut_rate, gen_gap, query, sim_treshold, porpouse, chemaxon=True):
        self.pop_size = int(pop_size)
        self.mut_rate = float(mut_rate)
        self.gen_gap = float(gen_gap)
        self.porpouse = porpouse

        if self.porpouse == 'linear' or self.porpouse == 'cyclic':
            self.B = ['']
            self.B4rndm = ['']
        if not os.path.exists(query):
            os.makedirs(query)
        self.folder = query
        self.query = self.interprete(query)

        if chemaxon:
            self.query_fp = self.calc_mxfp([self.query], 'Query')[1][0]
        else:
            self.query_fp = None
            self.fitness = 'Levenshtein'

        self.sim_treshold = int(sim_treshold)

    def rndm_seq(self):
        """Generates random implicit sequences of max "max_gen_no" generation dendrimers
           with max "max_aa_no" AA in each generation, picking from AA4random, B4random
           (probability of position to be empty intrinsic in these lists). 
        
        Returns:
            string -- implicit sequence of a random dendrimer
        """

        new_random_dendrimer = random.choice(self.CTrndm)
        aa_count = 0

        while aa_count < self.max_aa_no:
            new_random_dendrimer += random.choice(self.AA4rndm)
            aa_count += 1
        gen_count = 0
        while gen_count < self.max_gen_no:
            if new_random_dendrimer != '':
                new_random_dendrimer += random.choice(self.B4rndm)
            aa_count = 0
            while aa_count < self.max_aa_no:
                new_random_dendrimer += random.choice(self.AA4rndm)
                aa_count += 1
            gen_count += 1
        new_random_dendrimer += random.choice(self.NTrndm)

        return new_random_dendrimer[::-1]

    def rndm_gen(self):
        """Creates a generation of "pop_size" random dendrimers        
        Returns:
           list -- generation of "pop_size" random dendrimers
        """

        gen = []
        while len(gen) < self.pop_size:
            gen.append(self.rndm_seq())
        return gen

    def find_aa_b_pos(self, seq):
        """finds aminoacids and branching unit positions in a given sequence
        
        Arguments:
            seq {string} -- peptide dendrimer sequence
        
        Returns:
            lists -- aminoacids and branching units positions, all position, terminal pos, capping 
        """

        aa = []
        b = []
        all_pos = []
        met = []

        for i, symbol in enumerate(seq):
            if symbol in ['X', 'Ö', 'Ü', 'Ä', 'ö', 'ü', 'ä', '+', '&']:
                continue
            if symbol == '-':
                met.append(i)
                continue
            if symbol in self.B_SMILES.keys():
                b.append(i)
            elif symbol in self.AA_SMILES.keys():
                aa.append(i)
            all_pos.append(i)

        return aa, b, met, all_pos

    def split_seq_components(self, seq):
        """split seq in generations and branching units
    
        Arguments:
            seq {string} -- dendrimer sequence

        Returns:
            lists -- generations(gs, from 0 to..), branching units, terminal and capping
        """

        g = []
        gs = []
        bs = []
        t = []
        c = []

        for ix, i in enumerate(seq):
            if i not in ['1', '2', '3', '4', '5', '6', '7', '8']:
                if i in self.CT:
                    t.append(i)
                elif i in self.NT:
                    c.append(i)
                elif i == 'X':
                    continue
                elif i == '-':
                    if seq[ix - 1] in ['1', '2', '3', '4', '5', '6', '7', '8']:
                        bs.append(i)
                    else:
                        g.append(i)
                else:
                    g.append(i)
            else:
                gs.append(g[::-1])
                bs.append(i)
                g = []
        gs.append(g[::-1])
        gs = gs[::-1]
        bs = bs[::-1]

        return gs, bs, t, c

    def pick_aa_b_pos(self, seq, type_pos):
        """If type is aa, it returns an aminoacid position in the given sequence.
        if type is b, it returns a branching unit position in the given sequence.
        if type is all it returns a random position in the given sequence.
        
        Arguments:
            seq {string} -- peptide dendirmer sequence
            type_pos {string} -- aa, b or None
        
        Returns:
            int -- position
        """

        aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
        if type_pos == 'aa':
            return random.choice(aa_pos)
        elif type_pos == 'b':
            return random.choice(b_pos)
        elif type_pos == 'met':
            return random.choice(met_pos)
        elif type_pos == 'all':
            return random.choice(all_pos)
        else:
            if self.verbose:
                print('not valid type, type has to be "aa", "b" or "all"')

    def connect_mol(self, mol1, mol2):
        """it is connecting all Nterminals of mol1 with the Cterminal 
        of the maximum possible number of mol2s
    
        Arguments:
            mol1 {rdKit mol object} -- first molecule to be connected
            mol2 {rdKit mol object} -- second molecule to be connected

        Returns:
            rdKit mol object -- mol1 updated (connected with mol2, one or more)
        """
        count = 0

        # detects all the N terminals in mol1
        for atom in mol1.GetAtoms():
            atom.SetProp('Cterm', 'False')
            atom.SetProp('methyl', 'False')
            if atom.GetSmarts() == '[N:2]' or atom.GetSmarts() == '[NH2:2]' or atom.GetSmarts() == '[NH:2]':
                count += 1
                atom.SetProp('Nterm', 'True')
            else:
                atom.SetProp('Nterm', 'False')

        # detects all the C terminals in mol2 (it should be one)
        for atom in mol2.GetAtoms():
            atom.SetProp('Nterm', 'False')
            atom.SetProp('methyl', 'False')
            if atom.GetSmarts() == '[C:1]' or atom.GetSmarts() == '[CH:1]':
                atom.SetProp('Cterm', 'True')
            else:
                atom.SetProp('Cterm', 'False')

        # mol2 is addes to all the N terminal of mol1
        for i in range(count):
            combo = rdmolops.CombineMols(mol1, mol2)
            Nterm = []
            Cterm = []

            # saves in two different lists the index of the atoms which has to be connected
            for atom in combo.GetAtoms():
                if atom.GetProp('Nterm') == 'True':
                    Nterm.append(atom.GetIdx())
                if atom.GetProp('Cterm') == 'True':
                    Cterm.append(atom.GetIdx())

            # creates the amide bond
            edcombo = rdchem.EditableMol(combo)
            edcombo.AddBond(Nterm[0], Cterm[0], order=Chem.rdchem.BondType.SINGLE)
            edcombo.RemoveAtom(Cterm[0] + 1)
            clippedMol = edcombo.GetMol()

            # removes tags and lables form c term atoms which reacted
            clippedMol.GetAtomWithIdx(Cterm[0]).SetProp('Cterm', 'False')
            clippedMol.GetAtomWithIdx(Cterm[0]).SetAtomMapNum(0)

            # methylates amide bond
            if self.metbond == True and self.methyl == True:
                Nterm = []
                Met = []
                methyl = rdmolfiles.MolFromSmiles('[C:4]')
                for atom in methyl.GetAtoms():
                    atom.SetProp('methyl', 'True')
                    atom.SetProp('Nterm', 'False')
                    atom.SetProp('Cterm', 'False')
                metcombo = rdmolops.CombineMols(clippedMol, methyl)
                for atom in metcombo.GetAtoms():
                    if atom.GetProp('Nterm') == 'True':
                        Nterm.append(atom.GetIdx())
                    if atom.GetProp('methyl') == 'True':
                        Met.append(atom.GetIdx())
                metedcombo = rdchem.EditableMol(metcombo)
                metedcombo.AddBond(Nterm[0], Met[0], order=Chem.rdchem.BondType.SINGLE)
                clippedMol = metedcombo.GetMol()
                clippedMol.GetAtomWithIdx(Met[0]).SetProp('methyl', 'False')
                clippedMol.GetAtomWithIdx(Met[0]).SetAtomMapNum(0)

            # removes tags and lables form the atoms which reacted
            clippedMol.GetAtomWithIdx(Nterm[0]).SetProp('Nterm', 'False')
            clippedMol.GetAtomWithIdx(Nterm[0]).SetAtomMapNum(0)

            # uptades the 'core' molecule
            mol1 = clippedMol
        self.metbond = False
        return mol1

    def smiles_from_seq(self, seq):
        """Calculates the smiles of a given peptide dendrimer sequence
    
        Arguments:
            seq {string} -- peptide dendrimer sequence
        Returns:
            string -- molecule_smile - SMILES of the peptide
        """

        gs, bs, terminal, capping = self.split_seq_components(seq)

        # modifies the Cterminal
        if terminal:
            molecule = rdmolfiles.MolFromSmiles(self.T_SMILES[terminal[0]])
        else:
            molecule = ''

        # creates the dendrimer structure
        for gen in gs:
            for aa in gen:
                if aa == '-':
                    self.metbond = True
                    continue
                if molecule == '':
                    molecule = rdmolfiles.MolFromSmiles(self.AA_SMILES[aa])
                else:
                    molecule = self.connect_mol(molecule, rdmolfiles.MolFromSmiles(self.AA_SMILES[aa]))

            if bs:
                if bs[0] == '-':
                    self.metbond = True
                    bs.pop(0)
                if molecule == '':
                    molecule = rdmolfiles.MolFromSmiles(self.B_SMILES[bs[0]])
                else:
                    molecule = self.connect_mol(molecule, rdmolfiles.MolFromSmiles(self.B_SMILES[bs[0]]))
                bs.pop(0)

        # adds capping to the N-terminal (the called clip function is different, cause the listed smiles 
        # for the capping are already without OH, it is not necessary removing any atom after foming the new bond)
        if capping and molecule != '':
            molecule = attach_capping(molecule, rdmolfiles.MolFromSmiles(self.C_SMILES[capping[0]]))

        # clean the smile from all the tags
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)

        molecule_smile = rdmolfiles.MolToSmiles(molecule, isomericSmiles=True).replace('[N]', 'N').replace('[C]', 'C')
        return molecule_smile

    def smiles_from_seq_cyclic(self, seq):
        """Calculates the smiles of the given peptide sequence and cyclize it
    
        Arguments:
            seq {string} -- peptide dendrimer sequence
        Returns:
            string -- molecule_smile - SMILES of the peptide
        """

        if 'X' in seq:
            cy = 1
            for i in self.NT:
                seq = seq.replace(i, '')
            for i in self.CT:
                seq = seq.replace(i, '')
        else:
            cy = 0

        gs, bs, terminal, capping = self.split_seq_components(seq)

        # modifies the Cterminal
        if terminal:
            molecule = rdmolfiles.MolFromSmiles(self.T_SMILES[terminal[0]])
        else:
            molecule = ''

        if bs:
            if self.verbose:
                print('dendrimer, cyclization not possible, branching unit will not be considered')

        # creates the linear peptide structure
        for gen in gs:
            for aa in gen:
                if aa == 'X':
                    continue
                if aa == '-':
                    self.metbond = True
                    continue
                if molecule == '':
                    molecule = rdmolfiles.MolFromSmiles(self.AA_SMILES[aa])
                else:
                    molecule = self.connect_mol(molecule, rdmolfiles.MolFromSmiles(self.AA_SMILES[aa]))

        # adds capping to the N-terminal (the called clip function is different, cause the listed smiles 
        # for the capping are already without OH, it is not necessary removing any atom after foming the new bond)
        if capping:
            molecule = attach_capping(molecule, rdmolfiles.MolFromSmiles(self.C_SMILES[capping[0]]))

        # cyclize
        if molecule == '':
            smiles = ''
            return smiles, seq
        molecule = cyclize(molecule, cy)

        # clean the smile from all the tags
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = rdmolfiles.MolToSmiles(molecule, isomericSmiles=True).replace('[N]', 'N').replace('[C]', 'C')

        return smiles, seq

    def calc_mxfp(self, seqs, descriptor):
        """Calculates the mxfp for the given values
        
        Arguments:
            seqs {list} -- peptide dendrimer sequence list
            gen_n {string} -- for the characterization of the created mxfp file
        
        Returns:
            precessed sesq, fps, smiles, props {lists} -- processed sequences and the relative mxfp
        """

        with open('{}/{}_tmp.smi'.format(self.folder, descriptor), '+w') as outFile:
            for seq in seqs:
                if seq == '&' or seq == 'X' or seq == '+':
                    continue
                if self.porpouse == 'cyclic':
                    smi, seq = self.smiles_from_seq_cyclic(seq)
                else:
                    smi = self.smiles_from_seq(seq)
                if smi == '':
                    continue
                outFile.write(smi + ' ' + seq + ' a' + ' a' + ' a' + '\n')

        proc = sub.Popen(['java', '-cp',
                          '/home/alice/Code/PDGA/MXFP_peptides/2dpguassfp-master.jar',
                          'bin.write_topoguassfp2', '-i', '{}/{}_tmp.smi'.format(self.folder, descriptor),
                          '-o', '{}/{}_tmp.fp'.format(self.folder, descriptor),
                          '-scaleFactors 0.5_1_0.5_1_1_1_1'], stdout=sub.PIPE)

        proc.wait()
        fps = []
        proc_seqs = []
        smiles = []
        prop = []

        with open('{}/{}_tmp.fp'.format(self.folder, descriptor)) as inFile:
            for line in inFile:
                line = line.strip()
                line = line.split(' ')
                fps.append(line[2])
                smiles.append(line[0])
                proc_seqs.append(line[1])
                prop.append(line[3])

        if descriptor != 'Query':
            os.remove('{}/{}_tmp.smi'.format(self.folder, descriptor))
            os.remove('{}/{}_tmp.fp'.format(self.folder, descriptor))

        return proc_seqs, fps, smiles, prop

    def levenshtein(self, s1, s2):
        """calculates Levenshtein distance between seq1 and seq2
        
        Arguments:
            s1 {string} -- seq
            s2 {string} -- seq
        
        Returns:
            int -- Levenshtein distance
        """

        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[
                                 j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1  # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def mutate_aa(self, seq):
        """Performs n (mut_n, class variable) random point mutation

        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """
        mutations = []

        for i in range(self.mut_n):
            aa_pos = self.pick_aa_b_pos(seq, 'aa')
            seq1 = seq[:aa_pos]
            seq2 = seq[aa_pos + 1:]
            aa_new = np.random.choice(self.AA, 1)
            seq = seq1 + aa_new[0] + seq2
            mutations.append(seq)

        return mutations

    def mutate_b(self, seq):
        """Performs n (mut_n, class variable) random point mutation

        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """
        mutations = []

        for i in range(self.mut_n):
            b_pos = self.pick_aa_b_pos(seq, 'b')
            seq1 = seq[:b_pos]
            seq2 = seq[b_pos + 1:]
            b_new = np.random.choice(self.B, 1)
            seq = seq1 + b_new[0] + seq2
            mutations.append(seq)

        return mutations

    def move_b(self, seq, pos):
        """Performs n (mut_n, class variable) random point mutation

        Arguments:
            seq {string} -- seq to be mutated
            pos {integer} -- position to move the branching unit, positive for right, negative for left

        Returns:
            list -- mutations
        """

        mutations = []
        for i in range(self.mut_n):
            b_pos = self.pick_aa_b_pos(seq, 'b')
            b = seq[b_pos]
            if 0 <= b_pos + pos < len(seq):
                if seq[b_pos + pos] in self.CT or seq[b_pos + pos] in self.NT:
                    mutations.append(seq)
                    if self.verbose:
                        print(seq + ' Terminal found, could not move ' + b + ' {}'.format(pos))
                    continue
                else:
                    seqd = seq[:b_pos] + seq[b_pos + 1:]
                    seq1 = seqd[:b_pos + pos]
                    seq2 = seqd[b_pos + pos:]
                    seq = seq1 + b + seq2
                    mutations.append(seq)
            else:
                mutations.append(seq)

        return mutations

    def insert(self, seq, type_insert):
        """Performs n (mut_n, class variable) random point insertions. 
        If type insert is 'aa' the new element will be an aminoacid.
        If type insert is 'b' the new element will be a branching unit.
    
        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """

        mutations = []
        for i in range(self.mut_n):
            pos = self.pick_aa_b_pos(seq, 'all')
            if seq[pos] in self.NT or seq[pos] in self.CT or seq[pos] == 'X':
                mutations.append(seq)
                continue

            if type_insert == 'aa':
                new_element = np.random.choice(self.AA, 1)
            elif type_insert == 'b':
                new_element = np.random.choice(self.B, 1)
            else:
                raise ValueError("not valid type, type has to be \"aa\" or \"b\"")

            seq1 = seq[:pos]
            seq2 = seq[pos:]
            seq = seq1 + new_element[0] + seq2
            mutations.append(seq)

        return mutations

    def delete(self, seq):
        """Performs n (mut_n, class variable) deletion 

        Arguments:
            seq {string} -- seq to be mutated

        Returns:
            list -- mutations
        """

        mutations = []
        for i in range(self.mut_n):
            pos = self.pick_aa_b_pos(seq, 'all')
            seq1 = seq[:pos]
            seq2 = seq[pos + 1:]
            new_seq = seq1 + seq2
            mutations.append(new_seq)

        return mutations

    def fitness_function(self, gen):
        """Calculates the probability of survival of each seq in generation "gen"
    
        Arguments:
            gen {list} -- sequences
            gen_n {int} -- generation number

        Returns:
            cbd_av, cbd_min {int} -- average and minumum cbds of gen
            dist_dict, survival_dict {dict} -- {seq:cbd}, {seq:probability_of_survival}
        """

        dist_dict = {}
        gen_to_calc = []

        for seq in gen:
            if seq in self.dist_dict_old:
                dist_dict[seq] = self.dist_dict_old[seq]
            else:
                gen_to_calc.append(seq)

        if self.fitness == 'levenshtein' or self.fitness == 'Levenshtein':
            seqs = gen
            smiles = None
            prop = None
            mxfp = None
        else:
            seqs, fps, smiles_l, props = self.calc_mxfp(gen_to_calc, 'Gen{}'.format(self.gen_n))

        for i, seq in enumerate(seqs):
            if self.fitness == 'Levenshtein' or self.fitness == 'levenshtein':
                dist_dict[seq] = self.levenshtein(self.query, seq)
                if dist_dict[seq] <= self.sim_treshold:
                    self.write_results(smiles, seq, mxfp, prop, dist_dict[seq])
                continue

            mxfp = fps[i]
            smiles = smiles_l[i]
            prop = props[i]
            if mxfp is None or mxfp == 'a':
                continue
            cbd = calc_cbd(self.query_fp, mxfp)
            if cbd <= self.sim_treshold:
                self.write_results(smiles, seq, mxfp, prop, cbd)
            dist_dict[seq] = cbd

        survival_dict = {}

        for k, v in dist_dict.items():
            survival_dict[k] = 1 / (v + 1)

        survival_sum = sum(survival_dict.values())
        survival_dict = {k: (v / survival_sum) for k, v in survival_dict.items()}

        cbd_av = sum(dist_dict.values()) / len(dist_dict.values())
        cbd_min = min(dist_dict.values())

        # updates class variable dist_dict_old
        self.dist_dict_old = dist_dict
        return cbd_av, cbd_min, dist_dict, survival_dict

    def who_lives(self):
        """Returns the sequences that will remain unchanged
        
        Returns:
            list -- chosen sequences that will live
        """

        sorted_gen = sorted(self.surv_dict.items(), key=lambda x: x[1], reverse=True)
        if sorted_gen[0][0] not in self.min_dict.keys():
            self.min_dict[sorted_gen[0][0]] = self.gen_n
        fraction = int((1 - self.gen_gap) * self.pop_size)
        if len(list(self.surv_dict.keys())) <= fraction:
            return list(self.surv_dict.keys())
        else:
            wholives = []
            if self.selec_strategy == 'Elitist':
                for element in range(fraction):
                    wholives.append(sorted_gen[element][0])
                return wholives
            elif self.selec_strategy == 'Pure':
                while len(wholives) < fraction:
                    new = np.random.choice(list(self.surv_dict.keys()), 1, p=list(self.surv_dict.values()))[0]
                    if new not in wholives:
                        wholives.append(new)
                return wholives
            else:
                if self.verbose:
                    print('not valid selection strategy, type has to be "Elitist", or "Pure"')

    def pick_parents(self):
        """Picks two sequences according to their survival probabilities

        Arguments:
            surv_dict {dict} -- {sequence:survival_probability}

        Returns:
            list -- parents
        """

        parents = np.random.choice(list(self.surv_dict.keys()), 2, p=list(self.surv_dict.values()))
        return parents

    def make_new_gen(self, n):
        """Generates a new generation of n sequences with mating + 2 random sequences

        Arguments:
            n {int} -- number of structure to generate

        Returns:
            list -- new generation
        """

        new_gen = []

        for i in range(int(self.pop_size / self.rndm_newgen_fract)):
            new_gen.append(self.rndm_seq())

        while len(new_gen) < n:
            parents = self.pick_parents()
            child = mating(parents)
            new_gen.append(child)

        return new_gen

    def make_new_gen_cyclic(self, n):
        """Generates a new generation of n sequences with mutation

        Arguments:
            n {int} -- number of structure to generate

        Returns:
            list -- new generation
        """

        new_gen = []
        for i in range(int(self.pop_size / self.rndm_newgen_fract)):
            new_gen.append(self.rndm_seq())

        while len(new_gen) < n / 2:
            parents = self.pick_parents()

            child = mating(parents)

            if 'Ä' in child:
                child = child.replace('Ä', '')
            if 'Ö' in child:
                child = child.replace('Ö', '')
            if 'Ü' in child:
                child = child.replace('Ü', '')

            new_gen.append(child)

        while len(new_gen) < n:
            new_gen.append(np.random.choice(list(self.surv_dict.keys()), 1, p=list(self.surv_dict.values()))[0])

        new_gen_cy = self.mutate_cyclic(new_gen)

        return new_gen_cy

    def mutate(self, gen):
        """Mutates the given generation 

        Arguments:
            gen {list} -- sequences


        Returns:
            [list] -- mutated generation
        """

        mutations = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11']

        mutation = np.random.choice(mutations, 1, replace=False)

        gen_tmp = []

        if mutation == 'M1':
            seq_deletion = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_deletion:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen_tmp.append(self.delete(seq)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M2':
            seq_insertion_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen: 
                if seq in seq_insertion_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen_tmp.append(self.insert(seq, 'aa')[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M3':
            # to avoid incontrolled progressivly growth of the sequences,
            # mantain b_insert_rate (class variable, default = 0.1) low
            seq_insertion_b = np.random.choice(gen, int(round(len(gen) * self.b_insert_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_insertion_b:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen_tmp.append(self.insert(seq, 'b')[0])                    
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M4':
            seqs_mutate_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if aa_pos:
                        gen_tmp.append(self.mutate_aa(seq)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M5':
            seq_move_b_r = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_move_b_r:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if b_pos:
                        gen_tmp.append(self.move_b(seq, +1)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M6':
            seq_move_b_l = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_move_b_l:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if b_pos:
                        gen_tmp.append(self.move_b(seq, -1)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M7':
            seqs_mutate_b = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_b:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if b_pos:
                        gen_tmp.append(self.mutate_b(seq)[0])  
                    else:
                        gen_tmp.append(seq)           
                else:
                    gen_tmp.append(seq)

        if mutation == 'M8':
            seqs_mutate_c = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_c:
                    c = np.random.choice(self.NT, 1, replace=False)[0]
                    if len(seq) > 2 and seq[0] in self.NT:
                        seq_tmp = seq[1:]
                        new_seq = c + seq_tmp
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M9':
            seqs_mutate_t = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_t:
                    t = np.random.choice(self.CT, 1, replace=False)[0]
                    if len(seq) > 2 and seq[-1] in self.CT:
                        seq_tmp = seq[:-1]
                        new_seq = seq_tmp + t
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if self.methyl and mutation == 'M10':
            seqs_methylate = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_methylate:
                    gen_tmp.append(self.methylate(seq))
                else:
                    gen_tmp.append(seq)

        if self.methyl and mutation == 'M11':
            seqs_demethylate = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_demethylate:
                    gen_tmp.append(self.demethylate(seq))
                else:
                    gen_tmp.append(seq)        
        
        if gen_tmp == []:
            gen_tmp = gen
        gen_new = []

        for seq in gen_tmp:
            if seq == '':
                continue
            gen_new.append(seq)

        if self.methyl:
            for seq in gen_tmp:
                if '--' in seq:
                    new_seq = seq.replace('--', '-')
                    gen_new.append(new_seq)
                else:
                    gen_new.append(seq)

        return gen_new

    def methylate(self, seq):
        pos = self.pick_aa_b_pos(seq, 'all')
        if pos != (len(seq) - 1) and seq[pos + 1] != '-':
            new_seq = seq[:pos] + '-' + seq[pos:]
            seq = new_seq
        return seq

    def demethylate(self, seq):
        if '-' in seq:
            pos = self.pick_aa_b_pos(seq, 'met')
            new_seq = seq[:pos] + seq[pos + 1:]
            seq = new_seq
        return seq

    def form_SS(self, seq):
        """insertion of two ativated cys
        
        Arguments:
            seq {string} -- peptide seq
        
        Returns:
            string -- S-S cyclized peptide seq
        """

        act_cys = 'Ä'
        if 'Ä' in seq:
            act_cys = 'Ö'
            if 'Ö' in seq:
                act_cys = 'Ü'
                if 'Ü' in seq:
                    return seq

        if len(seq.replace('X', '').replace('-', '')) <= 2:
            return seq

        # first active cys
        pos = self.pick_aa_b_pos(seq, 'aa')
        seq_tmp = seq[:pos] + act_cys + seq[pos:]

        # second active cys
        pos = self.pick_aa_b_pos(seq, 'aa')
        new_seq = seq_tmp[:pos] + act_cys + seq_tmp[pos:]

        # prevents to activated cys next to each other
        if act_cys + act_cys not in new_seq:
            seq = new_seq

        return seq

    def mutate_cyclic(self, gen):
        """Mutates the given generation 

        Arguments:
            gen {list} -- sequences

        Returns:
            [list] -- mutated generation
        """

        mutations = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']

        mutation = np.random.choice(mutations, 1, replace=False)

        gen_tmp = []

        if mutation == 'M1':
            seq_deletion = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_deletion:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        new_seq = self.delete(seq)[0]
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M2':
            seq_insertion_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seq_insertion_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if all_pos:
                        gen.append(self.insert(seq, 'aa')[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M3':
            seqs_mutate_aa = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_aa:
                    aa_pos, b_pos, met_pos, all_pos = self.find_aa_b_pos(seq)
                    if aa_pos:
                        gen_tmp.append(self.mutate_aa(seq)[0])
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M4':
            seqs_mutate_c = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_c:
                    if 'X' not in seq:
                        c = np.random.choice(self.NT, 1, replace=False)[0]
                        if len(seq) > 2 and seq[0] in self.NT:
                            seq = seq[1:]
                            new_seq = c + seq
                            gen_tmp.append(new_seq)
                        else:
                            gen_tmp.append(seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)
                    
        if mutation == 'M5':
            seqs_mutate_t = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_mutate_t:
                    if 'X' not in seq:
                        t = np.random.choice(self.CT, 1, replace=False)[0]
                        if len(seq) > 2 and seq[-1] in self.CT:
                            seq = seq[:-1]
                            new_seq = seq + t
                            gen_tmp.append(new_seq)
                        else:
                            gen_tmp.append(seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)                            

        if mutation == 'M6':
            # break S-S
            seqs_inact_cys = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_inact_cys:
                    gen_tmp.append(break_SS(seq))
                else:
                    gen_tmp.append(seq)

        if mutation == 'M7':
            # make S-S
            seqs_act_cys = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_act_cys:
                    act_seq = self.form_SS(seq)
                    gen_tmp.append(act_seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M8':
            # linearize
            seqs_lin = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_lin:
                    if 'X' in seq:
                        new_seq = seq[1:]
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M9':
            # cyclize
            seqs_cy = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_cy:
                    if 'X' not in seq:
                        new_seq = 'X' + seq
                        for i in self.NT:
                            new_seq = new_seq.replace(i, '')
                        for i in self.CT:
                            new_seq = new_seq.replace(i, '')
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if mutation == 'M10':
            # swap head-to-tail with S-S
            seqs_swap = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_swap:
                    if 'X' in seq:
                        new_seq = swapcy(seq)
                        gen_tmp.append(new_seq)
                    else:
                        gen_tmp.append(seq)
                else:
                    gen_tmp.append(seq)

        if self.methyl and mutation == 'M11':
            seqs_methylate = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_methylate:
                    gen_tmp.append(self.methylate(seq))
                else:
                    gen_tmp.append(seq)

        if self.methyl and mutation == 'M12':
            seqs_demethylate = np.random.choice(gen, int(round(len(gen) * self.mut_rate, 0)), replace=False)
            for seq in gen:
                if seq in seqs_demethylate:
                    gen_tmp.append(self.demethylate(seq))
                else:
                    gen_tmp.append(seq)

        if gen_tmp == []:
            gen_tmp = gen
        gen_new = []

        for seq in gen_tmp:
            if seq == 'X' or seq == '' or seq == '+' or seq == '&':
                continue
            else:
                gen_new.append(seq)

        for seq in gen_tmp:
            for i in ['ÄÄ', 'ÖÖ', 'ÜÜ']:
                if i in seq:
                    new_seq = seq.replace(i, '')
                    gen_new.append(new_seq)
                else:
                    gen_new.append(seq)

        for seq in gen_tmp:
            for i in self.NT:
                if i in seq and 'X' in seq:
                    new_seq = seq.replace(i, '')
                    gen_new.append(new_seq)
                else:
                    gen_new.append(seq)

        for seq in gen_tmp:
            for i in self.CT:
                if i in seq and 'X' in seq:
                    new_seq = seq.replace(i, '')
                    gen_new.append(new_seq)
                else:
                    gen_new.append(seq)
        
        if self.methyl:
            for seq in gen_tmp:
                if '--' in seq:
                    new_seq = seq.replace('--', '-')
                    gen_new.append(new_seq)
                else:
                    gen_new.append(seq)
        
        return gen_new

    def write_results(self, smiles, seq, mxfp, prop, cbd):
        """if cbd from query is smaller than similarity treshold
            (class variable), adds seq to results
        
        """
        if self.fitness == 'Levenshtein' or self.fitness == 'levenshtein':
            with open('{}/{}_results'.format(self.folder, self.reinterprete(self.query)), 'a') as outFile:
                outFile.write(self.reinterprete(seq) + ' ' + str(cbd) + '\n')
        else:
            with open('{}/{}_results'.format(self.folder, self.reinterprete(self.query)), 'a') as outFile:
                outFile.write(smiles + ' ' + self.reinterprete(seq) + ' ' + mxfp + ' ' + prop + ' ' + str(cbd) + '\n')

    def write_progress(self):
        """add gen number, gen sequences and its CBD av and min
        
        """

        gen_temp = []
        gen = list(self.dist_dict.keys())
        for seq in gen:
            gen_temp.append(self.reinterprete(seq))
        gen = ';'.join(map(str, gen_temp))
        with open('{}/{}_generations'.format(self.folder, self.reinterprete(self.query)), 'a') as outFile:
            outFile.write(str(self.gen_n) + ' ' + gen + ' ' + str(self.cbd_av) + ' ' + str(self.cbd_min) + '\n')

    def write_param(self):
        with open('{}/param.txt'.format(self.folder), '+w') as outFile:
            outFile.write(str(self.__dict__) + '\n')
            outFile.write('Class variables: ' + '\n')
            outFile.write('used AA: ' + str(self.AA) + '\n')
            outFile.write('number of point mutation: ' + str(self.mut_n) + '\n')
            outFile.write('insert branching unit rate (mutation): ' + str(self.b_insert_rate) + '\n')
            outFile.write('survival strategy: ' + str(self.selec_strategy) + '\n')
            outFile.write('fraction of new generation that is random: ' + str(self.rndm_newgen_fract) + '\n')

    def set_verbose_true(self):
        """set verbose true (default false)
        """

        self.verbose = True

    def set_verbose_false(self):
        """set verbose false (default false)
        """

        self.verbose = False

    def exclude_buildingblocks(self, bb_to_ex):
        """Excludes the given building blocks
        
        Arguments:
            bb_to_ex {list} -- building blocks to exclude
        """

        for bb in bb_to_ex:
            if bb in self.interprete_dict.keys():
                element = self.interprete(bb)
                if element in self.AA:
                    self.exclude_aminoacids(element)
                elif element in self.B:
                    self.exclude_branching(element)
                elif element in self.CT:
                    self.exclude_C_terminal(element)
                elif element in self.NT:
                    self.exclude_N_capping(element)                
                elif bb == 'met':
                    self.exclude_methylation()
                else:
                    print("can't exclude ", bb)
            else:
                print("can't exclude ", bb)


    def exclude_aminoacids(self, aa_to_ex):
        """Excludes the given aminoacids
        
        Arguments:
            aa_to_ex {list} -- aminoacids to exclude
        """

        for element in aa_to_ex:
            self.AA.remove(element)
            self.AA4rndm.remove(element)
            self.AA4rndm.remove('')

        if not self.AA:
            self.AA.append('')
        if not self.AA4rndm:
            self.AA4rndm.append('')

        if self.verbose:
            print('The GA is using aminoacids:', self.AA)

    def exclude_branching(self, bb_to_ex):
        """Excludes the given branching units
        
        Arguments:
            bb_to_ex {list} -- branching units to exclude
        """

        if self.porpouse != 'cyclic':

            for element in bb_to_ex:
                self.B.remove(element)
                self.B4rndm.remove(element)

            if not self.B:
                self.B.append('')
            if not self.B4rndm:
                self.B4rndm.append('')

            if self.verbose:
                print('The GA is using branching units:', self.B)

    def exclude_methylation(self):
        """excludes the possibility of amide bond methylation
        """
        self.methyl = False

    def allow_methylation(self):
        """allows the possibility of amide bond methylation
        """
        self.methyl = True

    def exclude_C_terminal(self, t_to_ex):
        """Excludes the given C terminal modifications
        
        Arguments:
            t_to_ex {list} -- C terminal modifications to exclude
        """

        for element in t_to_ex:
            self.CT.remove(element)
            self.CTrndm.remove(element)
            self.CTrndm.remove('')

        if not self.CT:
            self.CT.append('')
        if not self.CTrndm:
            self.CTrndm.append('')

        if self.verbose:
            print('The GA is using C terminal mod:', self.CT)

    def exclude_N_capping(self, c_to_ex):
        """Excludes the given N terminal capping
        
        Arguments:
            c_to_ex {list} -- N terminal capping to exclude
        """

        for element in c_to_ex:
            self.NT.remove(element)
            self.NTrndm.remove(element)
            self.NTrndm.remove('')

        if not self.NT:
            self.NT.append('')
        if not self.NTrndm:
            self.NTrndm.append('')

        if self.verbose:
            print('The GA is using N capping mod:', self.NT)

    def set_time_limit(self, timelimit):
        """Sets the specified timelimit. 
        the GA will stop if the timelimit is reached even if the primary condition is not reached.
        
        Arguments:
            timelimit {string} -- hours:minutes:seconds
        """

        timelimit = timelimit.split(':')
        hours = int(timelimit[0])
        minutes = int(timelimit[1])
        seconds = int(timelimit[2])
        self.timelimit_seconds = int(seconds + minutes * 60 + hours * 3600)
        if self.verbose:
            print('The GA will stop after', timelimit[0], 'hours,', timelimit[1], 'minutes, and', timelimit[2],
                  'seconds')

    def print_time(self):
        """print running time
        """

        hours, rem = divmod(self.time, 3600)
        minutes, seconds = divmod(rem, 60)
        print('Time {:0>2}:{:0>2}:{:0>2}'.format(int(hours), int(minutes), int(seconds)))

    def interprete(self, seq):
        """translates from 3letters code to one symbol
        
        Arguments:
            seq {string} -- 3 letters code seq (e.g. Ala-Gly-Leu)
        
        Returns:
            string -- one letter symbol seq (e.g. AGL)
        """

        new_seq = ''
        seq = seq.split('-')
        for bb in seq:
            new_seq += self.interprete_dict[bb]
        seq = new_seq
        return seq

    def reinterprete(self, seq):
        """translates one symbol to three letters code
        
        Arguments:
            seq {string} -- one letter symbol seq (e.g. AGL)
        
        Returns:
            string -- 3 letters code seq (e.g. Ala-Gly-Leu)
        """

        new_seq = []
        for bb in seq:
            new_seq.append(self.interprete_rev_dict[bb])
        seq = '-'.join(new_seq)

        return seq

    def check(self, gen):
        gen_new = []

        for seq in gen:
            if seq == 'X' or seq == '' or seq == '+' or seq == '&':
                continue
            else:
                gen_new.append(seq)
        return gen_new

    def run(self):
        """Performs the genetic algorithm
 
        """
        startTime = time.time()

        # generation 0:
        gen = self.rndm_gen()

        if self.porpouse == 'cyclic':
            gen_cy = self.mutate_cyclic(gen)
            gen = gen_cy
        if self.verbose:
            print('Generation', self.gen_n)

        gen = self.check(gen)

        # fitness function and survival probability attribution:
        self.cbd_av, self.cbd_min, self.dist_dict, self.surv_dict = self.fitness_function(gen)

        if self.verbose:
            print('Average CBD =', self.cbd_av, 'Minimum CBD =', self.cbd_min)

        # progress file update (generations and their 
        # average and minimum CBD from query): 
        self.write_progress()

        self.time = int(time.time() - startTime)

        if self.verbose:
            self.print_time()

        # if query is found updates found identity count (class variable):
        if self.cbd_min == 0:
            self.found_identity += 1

            # updates generation number (class variable):
        self.gen_n += 1

        # default: GA runs for ten more generation after the query is found.
        while self.cbd_min != 0 or self.found_identity <= 10:

            if self.timelimit_seconds is not None and self.time > self.timelimit_seconds:
                if self.verbose:
                    print('time limit reached')
                break

            if self.verbose:
                print('Generation', self.gen_n)

            # the sequences to be kept intact are chosen:
            survivors = self.who_lives()

            # n. (pop size - len(survivors)) sequences 
            # are created with crossover or mutation (cyclic):
            if self.porpouse == 'cyclic':
                new_gen = self.make_new_gen_cyclic(self.pop_size - len(survivors))
            else:
                new_gen = self.make_new_gen(self.pop_size - len(survivors))

            # the next generation is the results of merging 
            # the survivors with the new sequences:
            if self.porpouse == 'cyclic':
                gen_merg = survivors + self.mutate_cyclic(new_gen)
            else:
                gen_merg = survivors + self.mutate(new_gen)

            # eventual duplicates are removed:
            gen = remove_duplicates(gen_merg)

            # fitness function and survival 
            # probability attribution:
            self.cbd_av, self.cbd_min, self.dist_dict, self.surv_dict = self.fitness_function(gen)
            if self.verbose:
                print('Average CBD =', self.cbd_av, 'Minumum CBD =', self.cbd_min)

            # progress file update (generations and their 
            # average and minimum CBD from query): 
            self.write_progress()

            self.time = int(time.time() - startTime)

            if self.verbose:
                self.print_time()

            # updates generation number (class variable):
            self.gen_n += 1

            # if query is found updates found identity count (class variable) 
            if self.cbd_min == 0:
                self.found_identity += 1
