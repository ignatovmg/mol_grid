from cython.view cimport array as cvarray
from libc.math cimport sqrt

import numpy as np


def grid_sum_spheres(atom_radius,
                     cell,
                     grid: np.array,
                     origin: np.array,
                     nodes_l: np.array,
                     nodes_u: np.array,
                     coords: np.array,
                     scale):
    coords = ((coords - origin) / cell)

    cdef size_t size = coords.shape[0]
    cdef int[:, :] cnodes_l = nodes_l
    cdef int[:, :] cnodes_u = nodes_u
    cdef double[:, :] ccoords = coords
    cdef double[:, :, :] cgrid = grid
    cdef double r2 = (atom_radius / cell)**2

    cdef int[:] l, u
    cdef double[:] c
    cdef int x, y, z
    cdef double cur_r2
    cdef double cscale = scale

    for i in range(size):
        l = cnodes_l[i, :]
        u = cnodes_u[i, :]
        c = ccoords[i, :]
        for x in range(l[0], u[0]):
            for y in range(l[1], u[1]):
                for z in range(l[2], u[2]):
                    cur_r2 = (x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2])
                    if r2 > cur_r2:
                        cgrid[x, y, z] += cscale


def grid_fill_spheres(atom_radius,
                      cell,
                      grid: np.array,
                      origin: np.array,
                      nodes_l: np.array,
                      nodes_u: np.array,
                      coords: np.array,
                      scale):
    coords = ((coords - origin) / cell)

    cdef size_t size = coords.shape[0]
    cdef int[:, :] cnodes_l = nodes_l
    cdef int[:, :] cnodes_u = nodes_u
    cdef double[:, :] ccoords = coords
    cdef double[:, :, :] cgrid = grid
    cdef double r2 = (atom_radius / cell)**2

    cdef int[:] l, u
    cdef double[:] c
    cdef int x, y, z
    cdef double cur_r2
    cdef double cscale = scale

    for i in range(size):
        l = cnodes_l[i, :]
        u = cnodes_u[i, :]
        c = ccoords[i, :]
        for x in range(l[0], u[0]):
            for y in range(l[1], u[1]):
                for z in range(l[2], u[2]):
                    cur_r2 = (x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2])
                    if r2 > cur_r2:
                        cgrid[x, y, z] = cscale


def grid_sum_boxes(grid: np.array,
                   nodes_l: np.array,
                   nodes_u: np.array,
                   scale):

    cdef size_t size = nodes_l.shape[0]
    cdef int[:, :] cnodes_l = nodes_l
    cdef int[:, :] cnodes_u = nodes_u
    cdef double[:, :, :] cgrid = grid
    cdef int[:] l, u;
    cdef int x, y, z
    cdef double cscale = scale

    for i in range(size):
        l = cnodes_l[i, :]
        u = cnodes_u[i, :]
        for x in range(l[0], u[0]):
            for y in range(l[1], u[1]):
                for z in range(l[2], u[2]):
                    cgrid[x, y, z] += cscale


def grid_fill_boxes(grid: np.array,
                    nodes_l: np.array,
                    nodes_u: np.array,
                    scale):

    cdef size_t size = nodes_l.shape[0]
    cdef int[:, :] cnodes_l = nodes_l
    cdef int[:, :] cnodes_u = nodes_u
    cdef double[:, :, :] cgrid = grid
    cdef int[:] l, u;
    cdef int x, y, z
    cdef double cscale = scale

    for i in range(size):
        l = cnodes_l[i, :]
        u = cnodes_u[i, :]
        for x in range(l[0], u[0]):
            for y in range(l[1], u[1]):
                for z in range(l[2], u[2]):
                    cgrid[x, y, z] = cscale


def _tab_exp(radius, step, scale):
    sgm = radius / 2
    sgm2 = sgm * sgm
    scl = scale / (sgm * np.sqrt(3.1415926535 * 2))
    exp_tab = np.arange(0, 2 * radius + step, step)  # 2 just in case
    exp_tab = scl * np.exp(-(exp_tab * exp_tab) / sgm2 * 0.5)
    #exp_tab = np.array([scl*np.exp(-(x*x/sgm2)*0.5) for x in np.arange(0, radius + step, step)])
    return exp_tab


def grid_sum_gaussians(atom_radius,
                       cell,
                       grid: np.array,
                       origin: np.array,
                       nodes_l: np.array,
                       nodes_u: np.array,
                       coords: np.array,
                       scale):
    coords = ((coords - origin) / cell)

    cdef double step = 0.01
    cdef double[:] exp_tab = _tab_exp(atom_radius, step, scale)

    cdef size_t size = coords.shape[0]
    cdef int[:, :] cnodes_l = nodes_l
    cdef int[:, :] cnodes_u = nodes_u
    cdef double[:, :] ccoords = coords
    cdef double[:, :, :] cgrid = grid
    cdef double r2 = (atom_radius / cell)**2

    cdef int[:] l, u
    cdef double[:] c
    cdef int x, y, z
    cdef double cur_r2

    for i in range(size):
        l = cnodes_l[i, :]
        u = cnodes_u[i, :]
        c = ccoords[i, :]
        for x in range(l[0], u[0]):
            for y in range(l[1], u[1]):
                for z in range(l[2], u[2]):
                    cur_r2 = (x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2])
                    if r2 > cur_r2:
                        cgrid[x, y, z] += exp_tab[<int>(sqrt(cur_r2) / step)]


