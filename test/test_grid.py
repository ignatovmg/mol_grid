import pytest

import numpy as np
import torch
import prody

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


