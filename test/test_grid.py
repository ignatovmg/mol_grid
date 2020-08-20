import pytest

import numpy as np
import torch
import prody
from itertools import product


from mol_grid import Grid, dot_torch


def test_create_1():
    g = Grid(np.ones((1, 3, 3, 3)), origin=np.ones(3), delta=np.ones(3))
    assert len(g.grid.shape) == 4


def test_create_2():
    with pytest.raises(RuntimeError):
        g = Grid(np.ones((1, 3, 3, )), origin=np.ones(3), delta=np.ones(3))


def test_create_3():
    with pytest.raises(RuntimeError):
        g = Grid(np.ones((1, 3, 3, 3)), origin=np.ones(3), delta=np.ones(2))


def test_create_4():
    with pytest.raises(RuntimeError):
        g = Grid(torch.ones((1, 3, 3, 3)), origin=torch.ones(3), delta=torch.ones(3))


def test_save(tmpdir):
    g = Grid(np.ones((2, 3, 3, 3)), origin=np.ones(3), delta=np.ones(3), names=['test1', 'test2'])
    g.save(tmpdir / 'grid.list')
    new = Grid(grid_list=tmpdir / 'grid.list')
    assert np.all(new.grid - g.grid < 0.0001)
    assert all([x == y for x, y in zip(new.names, g.names)])


def test_copy():
    g = Grid(np.ones((2, 3, 3, 3)), origin=np.ones(3), delta=np.ones(3), names=None)
    g_copy = g.copy()


def test_dot_noshift():
    g1 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 1, 1))
    g2 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 1, 1))
    output = dot_torch(*g1, *g2)
    assert torch.all(output == 3**3)


def test_dot_shift1():
    g1 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 1, 1))
    g2 = torch.ones((5, 3, 3, 3)), torch.tensor((1, 1, 1)), torch.tensor((1, 1, 1))
    output = dot_torch(*g1, *g2)
    assert torch.all(output == 2**3)


def test_dot_shift2():
    g1 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 1, 1))
    g2 = torch.ones((5, 3, 3, 3)), torch.tensor((2.2, 2.4, 2.3)), torch.tensor((1, 1, 1))
    output = dot_torch(*g1, *g2)
    assert torch.all(output == 1)


def test_dot_shift3():
    g1 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 1, 1))
    g1[0][1][0, :2, :2] = 2
    g2 = torch.ones((5, 4, 4, 4)), torch.tensor((-2.2, -2.3, -2.3)), torch.tensor((1, 1, 1))
    output = dot_torch(*g1, *g2)
    assert torch.all(output - torch.tensor([8, 16, 8, 8, 8]) < 0.0001)


def test_dot_out_of_bounds():
    g1 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 1, 1))
    g2 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 6, 0)), torch.tensor((1, 1, 1))
    output = dot_torch(*g1, *g2)
    assert torch.all(output == 0)


def test_dot_different_deltas():
    g1 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 1, 1))
    g2 = torch.ones((5, 3, 3, 3)), torch.tensor((0, 0, 0)), torch.tensor((1, 2, 1))
    with pytest.raises(RuntimeError):
        output = dot_torch(*g1, *g2)


############ Test rotations #############


def _get_rotation_matrix_around_X(angle):
    rot_mat = np.asarray(
        [[1,             0,              0],
         [0, np.cos(angle), -np.sin(angle)],
         [0, np.sin(angle), np.cos(angle)]]
    )
    return rot_mat


def _get_rotation_matrix_around_Y(angle):
    rot_mat = np.asarray(
        [[ np.cos(angle), 0, np.sin(angle)],
         [ 0,             1,             0],
         [-np.sin(angle), 0, np.cos(angle)]]
    )
    return rot_mat


def _get_rotation_matrix_around_Z(angle):
    rot_mat = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0],
         [np.sin(angle),  np.cos(angle), 0],
         [0,              0,             1]]
    )
    return rot_mat


# maps entries on a plane to itself rotated by 90 degrees
INDEX_MAP_90 = [
        [[2, 0], [1, 0], [0, 0]],
        [[2, 1], [1, 1], [0, 1]],
        [[2, 2], [1, 2], [0, 2]]
    ]


# maps entries on a plane to itself rotated by 180 degrees
INDEX_MAP_180 = [
        [[2, 2], [2, 1], [2, 0]],
        [[1, 2], [1, 1], [1, 0]],
        [[0, 2], [0, 1], [0, 0]]
    ]


def test_rotate_identity():
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(np.eye(3), np.array([0, 0, 0]))
    np.testing.assert_array_equal(before, g.grid)


def test_rotate_90_around_X():
    rot_mat = _get_rotation_matrix_around_X(np.pi / 2)
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(rot_mat, np.array([0, 0, 0]))

    # flip entries to match the rotated grid
    manually_rotated = np.zeros_like(g.grid)
    for i in range(g.grid.shape[1]):
        for _k, _j in product(range(3), range(3)):
            k, j = INDEX_MAP_90[_k][_j]
            manually_rotated[0, i, j, k] = before[0, i, _j, _k]

    # compare rotated grids
    np.testing.assert_array_equal(manually_rotated, g.grid)


def test_rotate_90_around_Y():
    rot_mat = _get_rotation_matrix_around_Y(np.pi / 2)
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(rot_mat, np.array([0, 0, 0]))

    # flip entries to match the rotated grid
    manually_rotated = np.zeros_like(g.grid)
    for j in range(g.grid.shape[2]):
        for _i, _k in product(range(3), range(3)):
            i, k = INDEX_MAP_90[_i][_k]
            manually_rotated[0, i, j, k] = before[0, _i, j, _k]

    # compare rotated grids
    np.testing.assert_array_equal(manually_rotated, g.grid)


def test_rotate_90_around_Z():
    rot_mat = _get_rotation_matrix_around_Z(np.pi / 2)
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(rot_mat, np.array([0, 0, 0]))

    # flip entries to match the rotated grid
    manually_rotated = np.zeros_like(g.grid)
    for k in range(g.grid.shape[3]):
        for _i, _j in product(range(3), range(3)):
            j, i = INDEX_MAP_90[_j][_i]
            manually_rotated[0, i, j, k] = before[0, _i, _j, k]

    # compare rotated grids
    np.testing.assert_array_equal(manually_rotated, g.grid)


def test_rotate_180_around_X():
    rot_mat = _get_rotation_matrix_around_X(np.pi)
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(rot_mat, np.array([0, 0, 0]))

    # flip entries to match the rotated grid
    manually_rotated = np.zeros_like(g.grid)
    for i in range(g.grid.shape[1]):
        for _k, _j in product(range(3), range(3)):
            k, j = INDEX_MAP_180[_k][_j]
            manually_rotated[0, i, j, k] = before[0, i, _j, _k]

    # compare rotated grids
    np.testing.assert_array_equal(manually_rotated, g.grid)


def test_rotate_180_around_Y():
    rot_mat = _get_rotation_matrix_around_Y(np.pi)
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(rot_mat, np.array([0, 0, 0]))

    # flip entries to match the rotated grid
    manually_rotated = np.zeros_like(g.grid)
    for j in range(g.grid.shape[2]):
        for _i, _k in product(range(3), range(3)):
            i, k = INDEX_MAP_180[_i][_k]
            manually_rotated[0, i, j, k] = before[0, _i, j, _k]

    # compare rotated grids
    np.testing.assert_array_equal(manually_rotated, g.grid)


def test_rotate_180_around_Z():
    rot_mat = _get_rotation_matrix_around_Z(np.pi)
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(rot_mat, np.array([0, 0, 0]))

    # flip entries to match the rotated grid
    manually_rotated = np.zeros_like(g.grid)
    for k in range(g.grid.shape[3]):
        for _i, _j in product(range(3), range(3)):
            j, i = INDEX_MAP_180[_j][_i]
            manually_rotated[0, i, j, k] = before[0, _i, _j, k]

    # compare rotated grids
    np.testing.assert_array_equal(manually_rotated, g.grid)


def test_translate():
    g = Grid(np.arange(27).reshape((1,3,3,3)), np.array([0,0,0]), np.array([1,1,1]))
    before = g.grid.copy()
    g.move(np.eye(3), np.array([1, 1, 1]))

    manually_translated = np.zeros_like(g.grid)
    manually_translated[0, 1:, 1:, 1:] = before[0, :2, :2, :2]

    # compare rotated grids
    np.testing.assert_array_equal(manually_translated, g.grid)