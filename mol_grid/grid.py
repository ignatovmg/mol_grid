import numpy as np
import torch
from gridData import Grid as _Grid
from path import Path


class Grid(object):
    def __init__(self, grid=None, origin=None, delta=None, grid_list=None, names=None):
        if grid_list is None:
            if not isinstance(grid, np.ndarray):
                raise RuntimeError('`grid` must be or type numpy.ndarray')
            if len(grid.shape) != 4:
                raise RuntimeError('`grid` must have exactly 4 dimensions')
            self.grid = grid  # 4D

            if not isinstance(origin, np.ndarray):
                raise RuntimeError('`origin` must be or type numpy.ndarray')
            if len(origin.shape) != 1:
                raise RuntimeError('`origin` must have exactly 1 dimension')
            if len(origin) != 3:
                raise RuntimeError('`origin` must have size 3')
            self.origin = origin

            if not isinstance(delta, np.ndarray):
                raise RuntimeError('`delta` must be or type numpy.ndarray')
            if len(delta.shape) != 1:
                raise RuntimeError('`delta` must have exactly 1 dimension')
            if len(delta) != 3:
                raise RuntimeError('`delta` must have size 3')
            self.delta = delta

            self.names = names
            if names is not None:
                assert (len(names) == grid.shape[0])
        else:
            self.read_grids(grid_list)

    def copy(self):
        return Grid(self.grid.copy(),
                    self.origin.copy(),
                    self.delta.copy(),
                    names=None if self.names is None else [x for x in self.names])

    def save(self, list_path='grids.list', grid_ids=None):
        list_path = Path(list_path)
        grids_dx = {}
        with open(list_path, 'w') as f:
            f.write('%i\n' % self.grid.shape[0])
            for i, chan in enumerate(self.grid):
                if grid_ids is not None and i not in grid_ids:
                    continue
                if self.names is None:
                    name = list_path.basename().stripext() + f'.{i}.dx'
                else:
                    name = f'{self.names[i]}.dx'
                grid_dx = _Grid(grid=chan, origin=self.origin, delta=self.delta)
                grid_dx.export(list_path.dirname() / name)
                grids_dx[name] = name
                f.write(name + '\n')

    def read_grids(self, grid_list):
        with open(grid_list, 'r') as f:
            files = list(map(str.strip, f))[1:]

        names, grids, origin, delta = [], [], None, None
        for f in files:
            g = _Grid()
            g.load(Path(grid_list).dirname() / f)
            origin = g.origin
            delta = g.delta
            grids.append(g.grid)
            names.append(str(Path(f).basename().stripext()))

        grids = np.stack(grids)
        self.grid = np.array(grids)
        self.origin = np.array(origin)
        self.delta = np.array(delta)
        self.names = names

    def get_center(self) -> np.ndarray:
        return self.origin + np.array(self.grid.shape[1:]) * self.delta / 2

    def carve_box(self, lower, upper):
        lower = np.array(lower)
        upper = np.array(upper)

        origin = self.origin
        delta = self.delta
        grid = self.grid

        l = np.ceil((lower - origin) / delta).astype(int)
        u = np.ceil((upper - origin) / delta).astype(int)

        grid = grid[:, l[0]:u[0], l[1]:u[1], l[2]:u[2]].copy()
        origin = origin + delta * np.array(l)
        return Grid(grid, origin, delta)

    def move(self, rot_mat, tr_vec):
        """
        Rotate the grid around its center then translate
        """
        shape = np.array(self.grid.shape[-3:])

        # translation in cells
        tr_vec = tr_vec / self.delta

        # compute mapping to the original non-rotated grid
        i3d = np.transpose(np.indices(shape), axes=[1, 2, 3, 0]).reshape(-1, 3)
        center = ((shape - 1) / 2)
        i3d_orig = (np.matmul(i3d - tr_vec - center, rot_mat.T) + center).round().astype(int)

        # filter out those which are out of bounds
        cond = (((i3d_orig >= shape) + (i3d_orig < 0)).sum(axis=1) > 0)
        buf = (i3d_orig % shape).T

        # fill values in rotated grid
        cond = np.stack([cond]*self.grid.shape[0])
        rotated = np.where(cond, 0, self.grid[:, buf[0], buf[1], buf[2]])
        rotated = rotated.reshape(self.grid.shape)
        self.grid = rotated

        return self


def dot_numpy(grid1: Grid, grid2: Grid):
    raise NotImplementedError()


def dot_torch(grid1: torch.Tensor, origin1: torch.Tensor, delta1: torch.Tensor,
              grid2: torch.Tensor, origin2: torch.Tensor, delta2: torch.Tensor) -> torch.Tensor:

    if not torch.equal(delta1, delta2):
        raise RuntimeError('Grids must have the same deltas')

    if grid1.shape[0] != grid2.shape[0]:
        raise RuntimeError('Number or channels must be the same')

    device = grid2.device

    # move to the coordinate system of grid1
    new_origin = torch.round((origin2 - origin1).to(grid1.dtype) / delta1).type(torch.int32)
    bottom_left_1 = torch.zeros(3, dtype=new_origin.dtype, device=device)
    upper_right_1 = torch.tensor(grid1.shape[1:], dtype=new_origin.dtype, device=device)
    bottom_left_2 = new_origin
    upper_right_2 = new_origin + torch.tensor(grid2.shape[1:], dtype=new_origin.dtype, device=device)

    # if the grids don't overlap return zeros
    if torch.any(bottom_left_1.ge(upper_right_2)) or torch.any(bottom_left_2.ge(upper_right_1)):
        return torch.zeros(grid1.shape[0], dtype=grid1.dtype, device=device)

    # extract overlapping subgrids
    bl = torch.max(bottom_left_1, bottom_left_2)
    ur = torch.min(upper_right_1, upper_right_2)
    subgrid1 = grid1[:, bl[0]:ur[0], bl[1]:ur[1], bl[2]:ur[2]]

    bl -= new_origin
    ur -= new_origin
    subgrid2 = grid2[:, bl[0]:ur[0], bl[1]:ur[1], bl[2]:ur[2]]

    # compute dot product for each channel
    output = torch.zeros(grid1.shape[0], dtype=grid1.dtype, device=device)
    for i in range(grid1.shape[0]):
        output[i] = torch.dot(torch.flatten(subgrid1[i]), torch.flatten(subgrid2[i]))

    return output
