import mdtraj as md
import os
import tempfile
import prody
import math
import numpy as np


VDW_RADII = {'H': 1.2, 'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.8, 'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85}


def _load_traj(ag):
    h, tmp = tempfile.mkstemp(dir='.', suffix='.pdb')
    os.close(h)
    prody.writePDB(tmp, ag)
    traj = md.load_pdb(tmp, frame=0)
    os.remove(tmp)
    return traj


def calc_sasa(ag, normalize=True, change_radii: dict=None, probe_radius=0.14):
    traj = _load_traj(ag)
    vdw_radii = VDW_RADII.copy()
    if change_radii is not None:
        vdw_radii.update(change_radii)
    max_sasa = {k: 4 * math.pi * r * r * 0.01 for k, r in vdw_radii.items()}
    atom_sasa = md.shrake_rupley(traj, change_radii=change_radii, probe_radius=probe_radius)[0]
    if normalize:
        atom_sasa /= np.array([max_sasa[x] for x in ag.getElements()])
    return atom_sasa
