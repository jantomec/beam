import numpy as np

def newton_raphson(
    mesh: float,
    dof: bool,
    fload: float,
    res: float,
    iter0: bool
):
    if iter0 == false:
        for j in range(mesh.n_ele):
            print("ok")
