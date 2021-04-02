import prody
from path import Path
from mol_grid import calc_sasa

TEST_DATA = Path(__file__).abspath().dirname() / 'data'


def test_sasa_1():
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    sasa = calc_sasa(ag, probe_radius=0.001)
    assert abs(sasa[0] - 1.0) < 0.1


def test_sasa_2():
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    sasa = calc_sasa(ag, normalize=False, probe_radius=0.001)
    assert abs(sasa[0] - 4 * 3.14 * 0.155**2) < 0.1


def test_sasa_3():
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    sasa = calc_sasa(ag, normalize=False, change_radii={'N': 0.3}, probe_radius=0.001)
    assert abs(sasa[0] - 4 * 3.14 * 0.3**2) < 0.1
