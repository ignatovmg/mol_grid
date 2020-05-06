import prody
import numpy as np
import os
import tempfile
import itertools
import mdtraj as md

import pyximport
pyximport.install()
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
                 sasa_layer=4.0,
                 sasa_radius=2.0,
                 sasa_penalty=None,
                 fill_value=1,
                 vacuum_value=None,
                 gaussians=True,
                 type=None,
                 box_center=None,
                 box_size=None,
                 centering='com',
                 around_atom_group=None):

        self.cell = cell
        self.padding = padding
        self.atom_radius = atom_radius
        self.sasa_layer = sasa_layer
        self.sasa_radius = sasa_radius
        self.config = config
        self.fill_value = fill_value
        self.gaussians = gaussians
        self.vacuum_value = vacuum_value
        self.centering = centering

        if callable(self.config):
            self._selector = self.config
        elif self.config == 'Elements':
            self._selector = elements
        elif self.config == 'FuncGroups':
            self._selector = elements_and_fgroups
        else:
            raise RuntimeError('Wrong config name')

        if type == 'rec':
            self.sasa_penalty = 0
            self.centering = 'coe'
        elif type == 'lig':
            self.sasa_penalty = None
            self.centering = 'com'
        elif type is None:
            self.sasa_penalty = None
            self.centering = 'coe'
        else:
            raise ValueError(f'Wrong value for `type` ({type})')

        if sasa_penalty is not None:
            self.sasa_penalty = sasa_penalty

        self.box_center = box_center  # angstroms
        self.box_size = box_size  # angstroms
        if around_atom_group:  # TODO: i dont want to keep the atom group here, but this way we
            origin, size = self._get_box_from_ag(around_atom_group)  # end up recalculating origin and size later
            self.box_center = list(origin + size * cell / 2)
            self.box_size = list(size * cell)

        # ensure no atoms will be missed
        half_diag = self.cell * np.sqrt(3) / 2
        if half_diag > self.atom_radius:
            raise RuntimeError(
                f"Atomic radius ({self.atom_radius}) must be larger, than cell's half diagonal ({self.cell})")

    def get_params(self):
        params = self.__dict__.copy()
        params['config'] = str(self.config)

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

    def _get_box_from_ag_coe(self, ag: prody.AtomGroup):
        coords = ag.getCoords()
        cmax, cmin = coords.max(0), coords.min(0)
        coe = (cmax + cmin) / 2
        size = np.floor((cmax - cmin + 0.001) / self.cell) + 2 * np.ceil(self.padding / self.cell)
        origin = coe - size * self.cell / 2
        return origin.astype(float), size.astype(int) + 1

    def _get_box_from_ag_com(self, ag: prody.AtomGroup):
        coords = ag.getCoords()
        cmax, cmin = coords.max(0), coords.min(0)
        com = coords.mean(axis=0)
        shape = 2 * np.max([(cmax - com), (com - cmin)], axis=0)
        size = np.floor((shape + 0.001) / self.cell) + 2 * np.ceil(self.padding / self.cell)
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

    def _get_box_from_center_coords(self, center, size):
        center = np.array(center)
        size = np.array(size)
        origin = center - size / 2
        size = np.ceil(size / self.cell)
        return origin.astype(float), size.astype(int) + 1

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
                     value):
        assert (len(grid.shape) == 3)

        nodes_l, nodes_u, coords = self._calc_square_bounds_for_each_atom(grid.shape, origin, sel.getCoords())
        if self.gaussians:
            # logger.debug('Sum atoms as gaussians')
            cgrids.grid_sum_gaussians(self.atom_radius, self.cell, grid, origin, nodes_l, nodes_u, coords, value)
        else:
            # logger.debug('Sum atoms as spheres')
            cgrids.grid_sum_spheres(self.atom_radius, self.cell, grid, origin, nodes_l, nodes_u, coords, value)

    def _grid_fill_3d(self,
                      grid: np.array,
                      origin: np.array,
                      sel: prody.Selection or prody.AtomGroup,
                      value):
        assert (len(grid.shape) == 3)

        nodes_l, nodes_u, coords = self._calc_square_bounds_for_each_atom(grid.shape, origin, sel.getCoords())
        cgrids.grid_fill_spheres(self.atom_radius, self.cell, grid, origin, nodes_l, nodes_u, coords, value)

    def _grid_sum_4d(self,
                     grid: np.array,
                     origin: np.array,
                     sel: prody.Selection or prody.AtomGroup,
                     value):
        assert (len(grid.shape) == 4)

        nodes_l, nodes_u, coords = self._calc_square_bounds_for_each_atom(grid.shape[1:], origin, sel.getCoords())
        for d in range(grid.shape[0]):
            if self.gaussians:
                # logger.debug('Sum atoms as gaussians')
                cgrids.grid_sum_gaussians(self.atom_radius, self.cell, grid[d], origin, nodes_l, nodes_u, coords,
                                         value)
            else:
                # logger.debug('Sum atoms as spheres')
                cgrids.grid_sum_spheres(self.atom_radius, self.cell, grid[d], origin, nodes_l, nodes_u, coords,
                                       value)

    def _grid_fill_4d(self,
                      grid: np.array,
                      origin: np.array,
                      sel: prody.Selection or prody.AtomGroup,
                      value):
        assert (len(grid.shape) == 4)

        nodes_l, nodes_u, coords = self._calc_square_bounds_for_each_atom(grid.shape[1:], origin, sel.getCoords())
        r2 = (self.atom_radius / self.cell) ** 2
        coords = (coords - origin) / self.cell
        for l, u, c in zip(nodes_l, nodes_u, coords):
            for x, y, z in itertools.product(range(l[0], u[0]), range(l[1], u[1]), range(l[2], u[2])):
                if r2 > (x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]) + (z - c[2]) * (z - c[2]):
                    grid[:, x, y, z] = value

        # for d in range(grid.shape[0]):
        #    cgrid.grid_fill_spheres(self.atom_radius, self.cell, grid[d], origin, nodes_l, nodes_u, coords, value)

    @staticmethod
    def _load_traj(ag):
        h, tmp = tempfile.mkstemp(dir='.', suffix='.pdb')
        os.close(h)
        prody.writePDB(tmp, ag)
        traj = md.load_pdb(tmp, frame=0)
        os.remove(tmp)
        return traj

    def _select_core(self, ag):
        traj = self._load_traj(ag)
        sasa_atoms = md.shrake_rupley(traj)[0]
        sasa_serials = map(str, ag.getSerials()[sasa_atoms > 0])
        core = ag.select(f'not (within {self.sasa_layer} of serial {" ".join(sasa_serials)})')
        if core is None:
            logger.warning('No core atoms found in the protein')
            return
        return core

    def fill_core(self, data, origin, ag, value):
        # logger.debug(f'Filling protein core with {self.sasa_penalty}')
        core = self._select_core(ag)
        if core is not None:
            self._grid_fill_4d(data, origin, core, value)
        return data

    def _extract_subgrid(self, grid, origin, delta, center, size):
        center = np.array(center)
        size = np.array(size)
        delta = np.array(delta)
        shape = np.array(grid.shape)[1:]

        new_origin = (center - size / 2)

        lower = np.ceil((new_origin - origin) / self.cell).astype(int)
        lower[lower < 0] = 0

        upper = np.floor((new_origin - origin + size) / self.cell).astype(int)
        upper[upper > shape] = shape[upper > shape]

        if sum(upper - lower) < 1.0:
            raise RuntimeError("Selected area is outside of the grid boundaries")

        new_origin = origin + lower * delta
        new_grid = grid[:, lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        return new_grid, new_origin

    def make_grids(self, ag: prody.AtomGroup) -> Grid:
        '''
        '''
        if self.box_size is not None:
            if self.box_center is None:
                center = self._get_box_center(ag)
            else:
                center = self.box_center
            origin, shape = self._get_box_from_center_coords(center, self.box_size)
        else:
            origin, shape = self._get_box_from_ag(ag)
        delta = tuple([self.cell] * 3)

        channels = self._selector(ag)
        data = np.zeros([len(channels)] + list(shape))
        logger.debug(f'Making grids: origin={str(origin)}, shape={str(data.shape)}')

        for i, sel in enumerate(channels):
            #logger.debug(f'Filling {i}')
            if not sel is None:
                self._grid_sum_3d(data[i], origin, sel, self.fill_value)

        if self.vacuum_value is not None:
            data[data < 0.001] = self.vacuum_value

        if self.sasa_penalty is not None:
            logger.debug(f'Filling protein core with {self.sasa_penalty}')
            core = self._select_core(ag)
            if core is not None:
                self._grid_fill_4d(data, origin, core, self.sasa_penalty)

        if np.all(data == 0):
            logger.warning('All grid points are zero')

        return Grid(data, np.array(origin), np.array(delta))

    def generate_grids_and_save(self, pdb, list_path='grids.list') -> Grid:
        '''
        Generate and save grids, return a tuple (grids, origin, delta)
        '''
        ag = prody.parsePDB(pdb)
        grid = self.make_grids(ag)
        grid.save(list_path)
        return grid
