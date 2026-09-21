#!/usr/bin/env python
"""Generate gmsh prism meshes for the Firedrake prism smoke test.

Produces three files in this directory:

  prism_reference.msh one prism whose vertices are the FIAT reference prism.
                   Written by hand, because gmsh will not place a single prism
                   at chosen coordinates. Used to derive the closure
                   permutation of plan section 5.3.

  prism_slab.msh   an unstructured triangulation of a square, extruded into
                   prisms. All prism axes point along +z.

  prism_warped.msh the same topology, with the interior node coordinates moved
                   so that no prism is affine and the axes tilt. The mesh file
                   records no extrusion structure, so Firedrake must treat it
                   as a fully unstructured prism mesh.

Physical groups:
  volume 1        the prism cells
  surface 1       bottom (z = 0),  triangular facets
  surface 2       top    (z = H),  triangular facets
  surface 3       sides,           quadrilateral facets

Run:  python prism_meshes/make_prism_mesh.py
"""
import os
import sys
import numpy as np
import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
H = 0.6          # slab height
NLAYERS = 2      # prism layers
LC = 0.45        # in-plane target element size


def build(path, warp):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("prisms")

    # A unit square, triangulated by Delaunay. The in-plane connectivity is
    # unstructured; no two prisms share a common reference frame by construction.
    p = [gmsh.model.geo.addPoint(x, y, 0, LC)
         for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    lines = [gmsh.model.geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # recombine=True is required, and the name is misleading. It recombines
    # the SIDE surfaces into quadrilaterals. The base stays triangular, because
    # a triangle has nothing to recombine with. The result is prisms.
    # With recombine=False gmsh splits the side quads into triangles, and then
    # meshes the volume with tetrahedra instead. Verified: 156 tets, 0 prisms.
    ext = gmsh.model.geo.extrude([(2, surf)], 0, 0, H,
                                 numElements=[NLAYERS], recombine=True)
    vol = [e[1] for e in ext if e[0] == 3]
    top = [e[1] for e in ext if e[0] == 2][0]
    sides = [e[1] for e in ext if e[0] == 2][1:]

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, vol, 1)
    gmsh.model.addPhysicalGroup(2, [surf], 1)   # bottom
    gmsh.model.addPhysicalGroup(2, [top], 2)    # top
    gmsh.model.addPhysicalGroup(2, sides, 3)    # sides
    gmsh.model.mesh.generate(3)

    if warp:
        # Move every node. The map is a diffeomorphism of the slab, so the mesh
        # stays valid, but no prism is affine and every axis tilts differently.
        tags, coords, _ = gmsh.model.mesh.getNodes()
        c = np.array(coords).reshape(-1, 3)
        x, y, z = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
        c[:, 0] = x + 0.18 * z * np.sin(2.3 * y)
        c[:, 1] = y + 0.15 * z * np.cos(1.9 * x)
        c[:, 2] = z * (1.0 + 0.25 * np.sin(1.7 * x) * np.cos(2.1 * y))
        for t, xyz in zip(tags, c):
            gmsh.model.mesh.setNode(int(t), xyz.tolist(), [])

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(path)

    ntypes = {}
    for dim, tag in gmsh.model.getEntities(3):
        et, etags, _ = gmsh.model.mesh.getElements(dim, tag)
        for t, tags in zip(et, etags):
            ntypes[int(t)] = ntypes.get(int(t), 0) + len(tags)
    gmsh.finalize()
    return ntypes


# gmsh node order for a 6-node prism: bottom triangle, then top triangle.
# These six points are the FIAT reference prism, reordered into that convention.
REFERENCE_MSH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 1 0 1
6 0 1 1
$EndNodes
$Elements
1
1 6 2 1 1 1 2 3 4 5 6
$EndElements
"""


if __name__ == "__main__":
    ref = os.path.join(HERE, "prism_reference.msh")
    with open(ref, "w") as fh:
        fh.write(REFERENCE_MSH)
    print(f"{'prism_reference.msh':<20} 1 prism (hand written, FIAT reference coordinates)")
    for name, warp in (("prism_slab.msh", False), ("prism_warped.msh", True)):
        path = os.path.join(HERE, name)
        types = build(path, warp)
        # gmsh element type 6 is the 6-node prism.
        kind = f"{types[6]} prisms (gmsh type 6), no other cell type" \
            if set(types) == {6} else f"WRONG: {types}"
        print(f"{name:<20} 3D element types: {kind}")
    print("\nA pure-prism file matters. PETSc rewrites the cell type to")
    print("DM_POLYTOPE_TRI_PRISM_TENSOR when a file holds tetrahedra AND prisms")
    print("(see petsc/src/dm/impls/plex/plexgmsh.c:1755). That variant uses a")
    print("different cone ordering, so the permutation of plan section 5.3")
    print("would not apply.")
