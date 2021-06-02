import pytest
import numpy as np
import prody
from path import Path
from mol_grid import GridMaker

TEST_DATA = Path(__file__).abspath().dirname() / 'data'


def test_create():
    GridMaker()


def test_create_1():
    GridMaker(config='FuncGroups')


def test_create_2():
    with pytest.raises(RuntimeError):
        GridMaker(config='WrongConfigName')


def test_create_3():
    GridMaker(config=lambda x: x.select('all'))


def test_create_4():
    with pytest.raises(RuntimeError):
        GridMaker(cell=2.0, atom_radius=1.0)


def test_grid_1atom_shape_1():
    maker = GridMaker(cell=1.0, atom_radius=2.0, padding=3.0, config='Elements')
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '1atom.pdb'))
    assert np.all(g.grid.shape == np.array([4, 7, 7, 7]))
    assert np.all(g.origin == np.array([-3] * 3))


def test_grid_1atom_shape_2():
    maker = GridMaker(cell=1.0, atom_radius=2.0, padding=5.0, config='Elements')
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '1atom.pdb'))
    assert np.all(g.grid.shape == np.array([4, 11, 11, 11]))
    assert np.all(g.origin == np.array([-5] * 3))


def test_grid_1atom_shape_3():
    maker = GridMaker(cell=0.5, atom_radius=2.0, padding=3.0, config='Elements')
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '1atom.pdb'))
    assert np.all(np.array(g.grid.shape == np.array([4, 13, 13, 13])))
    assert np.all(g.origin == np.array([-3] * 3))


def test_grid_1atom_values():
    maker = GridMaker(cell=1.0, atom_radius=2.0, padding=3.0, config='Elements')
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '1atom.pdb'))
    assert np.all(g.grid[0] == 0)
    assert np.all(g.grid[2] == 0)
    assert np.all(g.grid[3] == 0)
    assert g.grid[1][3, 3, 3] == pytest.approx(0.398, 0.01)
    assert g.grid[1][3, 3, 2] == pytest.approx(0.241, 0.01)
    assert np.all(g.grid[1].T - g.grid[1] < 0.001)


def test_grid_2atom_shape_2():
    maker = GridMaker(cell=1.0, atom_radius=2.0, padding=3.0, config='Elements')
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '2atom.pdb'))
    assert np.all(g.grid.shape == np.array([4, 9, 9, 9]))
    assert np.all(g.origin == np.array([-3] * 3))
    assert np.all(g.grid[2] == 0)
    assert np.all(g.grid[3] == 0)
    assert g.grid[1][3, 3, 3] == pytest.approx(0.398, 0.01)
    assert g.grid[1][3, 3, 2] == pytest.approx(0.241, 0.01)
    assert g.grid[0][5, 5, 5] == pytest.approx(0.398, 0.01)
    assert g.grid[0][5, 5, 4] == pytest.approx(0.241, 0.01)
    assert np.all(g.grid[0][3:8, 3:8, 3:8] - g.grid[1][1:6, 1:6, 1:6] < 0.001)


def test_weights():
    maker = GridMaker(cell=1.0, atom_radius=2.0, config=lambda x: [x])
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '1atom.pdb'), weights=[2.0])
    print(g.grid)


def test_grid_1atom_values_point():
    maker = GridMaker(cell=1.0, padding=3.0, config='Elements', mode='point')
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '1atom.pdb'))
    assert g.grid[1, 3, 3, 3] == pytest.approx(1.0, 0.001)
    g.grid[1, 3, 3, 3] = 0
    assert np.all(g.grid < 0.00001)


def test_grid_1atom_values_point_out_of_bounds():
    maker = GridMaker(cell=1.0, box_origin=[-10, -10, -10], box_shape=[3,3,3], config='Elements', mode='point')
    g = maker.make_grids(prody.parsePDB(TEST_DATA / '1atom.pdb'))
    assert np.all(g.grid < 0.00001)


# Origin and shape calculation

def test_origin_shape_1():
    maker = GridMaker(cell=1.0, box_size=[10, 10, 10])
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    origin, shape = maker._get_origin_and_shape(ag)
    assert np.array_equal(shape, np.array([10, 10, 10]))
    assert np.array_equal(origin, np.array([-5, -5, -5]))


def test_origin_shape_2():
    maker = GridMaker(cell=1.25, box_size=[10, 10, 10])
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    origin, shape = maker._get_origin_and_shape(ag)
    assert np.array_equal(shape, np.array([8, 8, 8]))
    assert np.array_equal(origin, np.array([-5, -5, -5]))


def test_origin_shape_3():
    maker = GridMaker(cell=1.0, box_origin=[-5, -5, -5])
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    origin, shape = maker._get_origin_and_shape(ag)
    assert np.array_equal(shape, np.array([10, 10, 10]))
    assert np.array_equal(origin, np.array([-5, -5, -5]))


def test_origin_shape_4():
    maker = GridMaker(cell=1.25, box_shape=[8, 8, 8])
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    origin, shape = maker._get_origin_and_shape(ag)
    assert np.array_equal(shape, np.array([8, 8, 8]))
    assert np.array_equal(origin, np.array([-5, -5, -5]))


def test_origin_shape_5():
    maker = GridMaker(cell=1.0, box_center=[1, 1, 1])
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    origin, shape = maker._get_origin_and_shape(ag)
    assert np.array_equal(shape, np.array([8, 8, 8]))
    assert np.array_equal(origin, np.array([-3, -3, -3]))


def test_origin_shape_6():
    maker = GridMaker(cell=1.0, box_origin=[-10, -10, -10], box_shape=[20, 20, 20])
    ag = prody.parsePDB(TEST_DATA / '1atom.pdb')
    origin, shape = maker._get_origin_and_shape(ag)
    assert np.array_equal(shape, np.array([20, 20, 20]))
    assert np.array_equal(origin, np.array([-10, -10, -10]))
