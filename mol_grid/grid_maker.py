import prody
import numpy as np
import itertools

from . import cgrids
from .loggers import logger
from .amino_acids import FUNCTIONAL_GROUPS
from .grid import Grid


def elements(ag):
    selections = [ag.select(f'(not water) and element {ele}') for ele in ['C', 'N', 'O', 'S']]
    return selections


def elements_and_fgroups(ag):
    selections = [ag.select(f'(not water) and element {ele}') for ele in ['C', 'N', 'O', 'S']]
    for fgroup in ['hydroxyl', 'carboxyl', 'amide', 'methyl', 'aromatic', 'positive']:
        group = FUNCTIONAL_GROUPS[fgroup]
        group_sel = ' or '.join([f"(resname {res} name {' '.join(itertools.chain(*itertools.chain(*groups)))})"
                                 for res, groups in group.items()])
        selections.append(ag.select(group_sel))
    return selections


class GridMaker(object):
    def __init__(self,
                 cell=1.0,
                 padding=3.0,
                 atom_radius=2.0,
                 config='FuncGroups',
                 mode='gaussian',  # gaussian / sphere
                 box_origin=None,
                 box_center=None,
                 box_size=None,
                 box_shape=None,
                 centering='com',  # com / coe
                 around_atom_group=None):

        self.cell = cell
        self.padding = padding
        self.atom_radius = atom_radius
        self.config = config
        self.mode = mode
        self.centering = centering

        if callable(self.config):
            self._selector = self.config
        elif self.config == 'Elements':
            self._selector = elements
        elif self.config == 'FuncGroups':
            self._selector = elements_and_fgroups
        else:
            raise RuntimeError('Wrong config name')

        if self.mode not in ['gaussian', 'sphere', 'point']:
            raise RuntimeError('mode must be set to "gaussian" or" sphere" or "point"')

        self.box_origin = box_origin
        self.box_center = box_center  # angstroms
        self.box_size = box_size  # angstroms
        self.box_shape = box_shape

        if around_atom_group:  # TODO: i dont want to keep the atom group here, but this way we
            origin, size = self._get_box_from_ag(around_atom_group)  # end up recalculating origin and size later
            self.box_size = size
            self.box_origin = origin

        # ensure no atoms will be missed
        half_diag = self.cell * np.sqrt(3) / 2
        if half_diag > self.atom_radius:
            raise RuntimeError(
                f"Atomic radius ({self.atom_radius}) must be larger, than cell's half diagonal ({self.cell})")

    def _get_coe(self, ag: prody.AtomGroup):
        coords = ag.getCoords()
        cmax, cmin = coords.max(0), coords.min(0)
        coe = (cmax + cmin) / 2
        return coe

    def _get_com(self, ag: prody.AtomGroup):
        return ag.getCoords().mean(axis=0)

    def _get_box_center(self, ag: prody.AtomGroup):
        if self.centering == 'coe':
            return self._get_coe(ag)
        elif self.centering == 'com':
            return self._get_com(ag)
        else:
            raise RuntimeError(f'Wrong centering type ({self.centering})')

    def _get_box_size_from_ag_and_center(self, ag, center):
        coords = ag.getCoords()
        cmax, cmin = coords.max(0), coords.min(0)
        size = 2 * np.max([cmax - center, center - cmin], axis=0)
        size = np.floor((size + 0.001) / self.cell) + 2 * np.ceil(self.padding / self.cell)
        return size

    def _get_box_from_ag_coe(self, ag: prody.AtomGroup):
        coords = ag.getCoords()
        cmax, cmin = coords.max(0), coords.min(0)
        coe = (cmax + cmin) / 2
        size = self._get_box_size_from_ag_and_center(ag, coe)
        origin = coe - size * self.cell / 2
        return origin.astype(float), size.astype(int) + 1

    def _get_box_from_ag_com(self, ag: prody.AtomGroup):
        coords = ag.getCoords()
        com = coords.mean(axis=0)
        size = self._get_box_size_from_ag_and_center(ag, com)
        origin = com - size * self.cell / 2
        return origin.astype(float), size.astype(int) + 1

    def _get_box_from_ag(self, ag: prody.AtomGroup):
        if self.centering == 'coe':
            origin, size = self._get_box_from_ag_coe(ag)
        elif self.centering == 'com':
            origin, size = self._get_box_from_ag_com(ag)
        else:
            raise RuntimeError(f'Wrong centering type ({self.centering})')
        return origin, size

    def _calc_square_bounds_for_each_atom(self, shape, origin: np.array, coords: np.array):
        shape = np.array(shape)
        assert (shape.shape[0] == 3)

        lower = origin
        upper = origin + shape * self.cell
        in_bounds = np.all((coords >= lower) & (coords <= upper), axis=1)
        coords = coords[in_bounds]

        nodes_l = np.ceil((coords - origin - self.atom_radius) / self.cell).astype(np.int32)
        nodes_l[nodes_l < 0] = 0

        nodes_u = np.floor((coords - origin + self.atom_radius) / self.cell).astype(np.int32) + 1
        nodes_u[nodes_u[:, 0] > shape[0], 0] = shape[0]
        nodes_u[nodes_u[:, 1] > shape[1], 1] = shape[1]
        nodes_u[nodes_u[:, 2] > shape[2], 2] = shape[2]
        # nodes_u = np.where((nodes_l + 1) > nodes_u, nodes_l + 1, nodes_u)
        return nodes_l, nodes_u, coords

    def _grid_sum_3d(self,
                     grid: np.array,
                     origin: np.array,
                     sel: prody.Selection or prody.AtomGroup,
                     weights=None):
        assert (len(grid.shape) == 3)
        if weights is None:
            weights = np.ones(len(sel))

        nodes_l, nodes_u, coords = self._calc_square_bounds_for_each_atom(grid.shape, origin, sel.getCoords())
        if self.mode == 'gaussian':
            # logger.debug('Sum atoms as gaussians')
            cgrids.grid_sum_gaussians(self.atom_radius, self.cell, grid, origin, nodes_l, nodes_u, coords, weights)
        if self.mode == 'sphere':
            # logger.debug('Sum atoms as spheres')
            cgrids.grid_sum_spheres(self.atom_radius, self.cell, grid, origin, nodes_l, nodes_u, coords, weights)
        if self.mode == 'point':
            cgrids.grid_sum_points(self.cell, grid, np.array(grid.shape), origin, coords, weights)

    def _get_origin_and_shape(self, ag):
        shape = None
        origin = None

        if self.box_shape is not None:
            shape = np.array(self.box_shape).astype(int)
            assert len(shape) == 3
        elif self.box_size is not None:
            box_size = np.array(self.box_size).astype(float)
            assert len(box_size) == 3
            shape = np.ceil(box_size / self.cell).astype(int)

        if self.box_origin is not None:
            origin = np.array(self.box_origin).astype(float)
            assert len(origin) == 3
            if shape is None:
                box_center = self._get_box_center(ag)
                shape = np.ceil((box_center - origin) * 2 / self.cell).astype(int)
        elif self.box_center is not None:
            box_center = np.array(self.box_center).astype(float)
            assert len(box_center) == 3
            if shape is None:
                shape = np.ceil(self._get_box_size_from_ag_and_center(ag, box_center) / self.cell).astype(int)
            origin = box_center - shape * self.cell * 0.5

        if shape is not None:
            if origin is None:
                origin = self._get_box_center(ag) - shape * self.cell * 0.5
        elif origin is None:
            origin, shape = self._get_box_from_ag(ag)
        else:
            raise RuntimeError('Cannot be here')

        crd = ag.getCoords()
        if np.all(crd.min(0) > origin + shape * self.cell) or np.all(crd.max(0) < origin):
            logger.warning('Atom group is outside the box')

        return origin, shape

    def make_grids(self, ag: prody.AtomGroup, weights=None) -> Grid:
        origin, shape = self._get_origin_and_shape(ag)
        delta = np.array([self.cell] * 3)

        channels = self._selector(ag)
        data = np.zeros([len(channels)] + list(shape))
        logger.debug(f'Making grids: origin={str(origin)}, shape={str(data.shape)}')

        for i, sel in enumerate(channels):
            if not sel is None:
                self._grid_sum_3d(data[i], origin, sel, weights=weights)

        if np.all(data == 0):
            logger.warning('All grid points are zero')

        return Grid(data, origin, delta)
