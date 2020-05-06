# MolGrid #

Small library for creating 3D grids from pdb files

### Installation ###

Run
```
pip install git+https://bitbucket.org/ignatovmg/mol_grid.git
```


### Usage ###

Import classes    
```
from mol_grid import Grid, GridMaker, dot_torch
```   

Create a grid with      
```
grid = Grid(grid=np.ones(10, 3, 3, 3), origin=np.array([1., 2., 3.]), delta=np.array([1., 1., 1.]))
```

Save it in .dx format     
```
grid.save("grid.list")
```

Make grid from prody atom group     
```
ag = prody.parsePDB(prody.fetchPDB('1ao7'))
maker = GridMaker()
grid = maker.make_grids(ag)
grid.save("1ao7.list")
```