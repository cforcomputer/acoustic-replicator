"""Scene assembly helpers: part + pole -> BEM scatterer refined around a landing site."""
import numpy as np
from .acoustics import Scatterer, Field
from .geometry import VoxelPart, refine_near

POLE_RADIUS = 1.25e-3          # 2.5 mm rod
DEFAULT_LEVELS = [(3.5e-3, 0.8e-3), (1.8e-3, 0.4e-3), (0.8e-3, 0.2e-3), (0.45e-3, 0.11e-3)]


def frame_from_normal(n):
    """Rows (t1, t2, n): right-handed frame whose third axis is the outward normal."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(ref, n); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n])


def new_part(extent=20e-3, z_top=45e-3, pole_tip_z=0.0, res=0.2e-3):
    """Empty build volume with the pole hanging down to pole_tip_z."""
    part = VoxelPart([-extent, -extent, -extent], [extent, extent, z_top + 2e-3], res=res)
    part.add_capsule([0, 0, pole_tip_z + POLE_RADIUS], [0, 0, z_top + 5e-3], POLE_RADIUS)
    return part


def build_field(part: VoxelPart, array, target=None, h=1.4e-3, levels=DEFAULT_LEVELS,
                zmax=40e-3, n_chief=40, verbose=False):
    """Mesh the current part, refine around `target`, solve the BEM, return a Field."""
    mesh = part.to_mesh(h=h, zmax=zmax)
    if target is not None:
        mesh = refine_near(mesh, target, levels, project=part.project)
    sc = Scatterer(mesh, array, n_chief=n_chief, verbose=verbose)
    return Field(array, sc)
