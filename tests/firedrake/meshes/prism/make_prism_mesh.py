#!/usr/bin/env python
"""Generate the gmsh prism meshes for the Firedrake prism tests.

Produces fourteen files in this directory:

  prism_reference.msh one prism whose vertices are the FIAT reference prism.
                   Written by hand, because gmsh will not place a single prism
                   at chosen coordinates. Used to derive the closure
                   permutation of plan section 5.3.

  prism_reference_marked.msh
                   the same prism, with every facet in its own physical group,
                   so that a test can measure each facet separately.

  prism_slab.msh   an unstructured triangulation of a square, extruded into
                   prisms. All prism axes point along +z.

  prism_slab_mixed_marker.msh
                   the same mesh as prism_slab.msh, with different physical
                   groups. Surface 4 holds triangular AND quadrilateral facets.

  prism_warped.msh the same topology, with the interior node coordinates moved
                   so that no prism is affine and the axes tilt. The mesh file
                   records no extrusion structure, so Firedrake must treat it
                   as a fully unstructured prism mesh.

  prism_order_r0.msh, prism_order_r1.msh, prism_order_r2.msh
                   a refinement sequence for the convergence order test. Each
                   level halves the element size in every direction, so the
                   ratio of the mesh sizes is exactly 2 and the measured order
                   is not polluted by an uneven refinement. The base
                   triangulation is transfinite for that reason: a Delaunay
                   triangulation at half the target size does not halve the
                   in-plane element size, and the observed order then sits far
                   from the theoretical one. The warp of prism_warped.msh is
                   applied to every level, so no prism is affine.

  prism_interior_marked.msh
                   two boxes, x < 0 and x > 0, each extruded in two stages,
                   so the mesh has a marked interior surface of each facet
                   shape. For the interior facet integral (dS) tests.

  prism_interior_marked_scrambled.msh, prism_warped_scrambled.msh,
  prism_order_r0_scrambled.msh, prism_order_r1_scrambled.msh,
  prism_order_r2_scrambled.msh
                   a copy of the mesh without the suffix, with the node list
                   of each prism changed by one of the 6 orientation-preserving
                   prism symmetries. See scramble below. The geometry and the
                   physical groups do not change.

Physical groups of every file but the three below:
  volume 1        the prism cells
  surface 1       bottom (z = 0),  triangular facets
  surface 2       top    (z = H),  triangular facets
  surface 3       sides,           quadrilateral facets

Physical groups of prism_slab_mixed_marker.msh:
  volume 1        the prism cells
  surface 2       top    (z = H),  triangular facets
  surface 3       the sides x = 1, y = 1 and x = 0, quadrilateral facets
  surface 4       bottom (z = 0) and the side y = 0, triangular and
                  quadrilateral facets

Physical groups of prism_reference_marked.msh:
  volume 1        the prism
  surface 1       z = 0,     triangle
  surface 2       z = 1,     triangle
  surface 3       y = 0,     quadrilateral, area 1
  surface 4       x = 0,     quadrilateral, area 1
  surface 5       x + y = 1, quadrilateral, area sqrt(2)

Physical groups of prism_interior_marked.msh, the box [-0.5, 0.5]^2 x [0, 1]:
  volume 1        the prism cells
  surface 1       bottom (z = 0),  triangular facets
  surface 10      the interior plane z = 0.5, 136 triangular facets, area 1
  surface 20      the interior plane x = 0, 42 quadrilateral facets, area 1
  surface 30      surfaces 10 and 20 together, both facet shapes, area 2

Run:  python tests/firedrake/meshes/prism/make_prism_mesh.py
"""
import os
import random

import numpy as np
import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
H = 0.6          # slab height
NLAYERS = 2      # prism layers
LC = 0.45        # in-plane target element size

# The convergence sequence: (in-plane divisions per side, prism layers) per
# level. Every level doubles both, so the element size halves exactly.
ORDER_LEVELS = ((4, 2), (8, 4), (16, 8))


def build(path, warp, divisions=None, nlayers=NLAYERS, mixed_marker=False):
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

    if divisions is not None:
        # A transfinite base, for the convergence sequence only. "Alternate"
        # flips the diagonal from one quadrilateral to the next, so the
        # triangles do not all share one orientation.
        for line in lines:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, divisions + 1)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf, "Alternate")

    # recombine=True is required, and the name is misleading. It recombines
    # the SIDE surfaces into quadrilaterals. The base stays triangular, because
    # a triangle has nothing to recombine with. The result is prisms.
    # With recombine=False gmsh splits the side quads into triangles, and then
    # meshes the volume with tetrahedra instead. Verified: 156 tets, 0 prisms.
    ext = gmsh.model.geo.extrude([(2, surf)], 0, 0, H,
                                 numElements=[nlayers], recombine=True)
    vol = [e[1] for e in ext if e[0] == 3]
    top = [e[1] for e in ext if e[0] == 2][0]
    sides = [e[1] for e in ext if e[0] == 2][1:]

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, vol, 1)
    if mixed_marker:
        # sides[0] is the extrusion of lines[0], the side y = 0.
        gmsh.model.addPhysicalGroup(2, [surf, sides[0]], 4)   # bottom and y = 0
        gmsh.model.addPhysicalGroup(2, [top], 2)              # top
        gmsh.model.addPhysicalGroup(2, sides[1:], 3)          # other sides
    else:
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


def build_interior_marked(path):
    """Write prism_interior_marked.msh: two boxes with marked interior surfaces.

    Each box is a half of the unit square, extruded to z = 0.5 and then again
    to z = 1. The first extrusion leaves the triangles of z = 0.5 inside the
    mesh, and the shared side x = 0 leaves quadrilaterals inside the mesh.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("interior")
    geo = gmsh.model.geo
    points = {xy: geo.addPoint(*xy, 0, 0.15)
              for xy in ((-.5, -.5), (0, -.5), (.5, -.5), (.5, .5), (0, .5), (-.5, .5))}

    def line(a, b):
        return geo.addLine(points[a], points[b])

    mid = line((0, -.5), (0, .5))
    s1 = geo.addPlaneSurface([geo.addCurveLoop(
        [line((-.5, -.5), (0, -.5)), mid, line((0, .5), (-.5, .5)), line((-.5, .5), (-.5, -.5))])])
    s2 = geo.addPlaneSurface([geo.addCurveLoop(
        [line((0, -.5), (.5, -.5)), line((.5, -.5), (.5, .5)), line((.5, .5), (0, .5)), -mid])])
    vols, mids, walls = [], [], []
    for s in (s1, s2):
        # recombine=True gives prisms; see build.
        e1 = geo.extrude([(2, s)], 0, 0, .5, numElements=[3], recombine=True)
        e2 = geo.extrude([e1[0]], 0, 0, .5, numElements=[3], recombine=True)
        vols += [e1[1][1], e2[1][1]]
        mids.append(e1[0][1])
    geo.synchronize()
    # The surfaces extruded from the line x = 0 are the surfaces whose
    # bounding box has x = 0 at both ends.
    for dim, tag in gmsh.model.getEntities(2):
        b = gmsh.model.getBoundingBox(dim, tag)
        if abs(b[0]) < 1e-9 and abs(b[3]) < 1e-9:
            walls.append(tag)
    gmsh.model.addPhysicalGroup(3, vols, 1)
    gmsh.model.addPhysicalGroup(2, sorted(set(mids)), 10)
    gmsh.model.addPhysicalGroup(2, sorted(set(walls)), 20)
    gmsh.model.addPhysicalGroup(2, sorted(set(mids)) + sorted(set(walls)), 30)
    gmsh.model.addPhysicalGroup(2, [s1, s2], 1)
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(path)

    ntypes = {}
    for dim, tag in gmsh.model.getEntities(3):
        et, etags, _ = gmsh.model.mesh.getElements(dim, tag)
        for t, tags in zip(et, etags):
            ntypes[int(t)] = ntypes.get(int(t), 0) + len(tags)
    gmsh.finalize()
    return ntypes


# The 6 orientation-preserving symmetries of the gmsh prism node order: the
# 3 rotations of the triangle, each with or without the exchange of the two
# bases. The exchange also reflects the triangle, so the handedness of the
# cell does not change.
PRISM_ROTATIONS = ((0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4))
PRISM_FLIP = (3, 5, 4, 0, 2, 1)
PRISM_SYMMETRIES = tuple(s for r in PRISM_ROTATIONS
                         for s in (r, tuple(r[PRISM_FLIP[i]] for i in range(6))))

# The mesh to scramble, and the seed, of each scrambled mesh.
SCRAMBLED = (("prism_interior_marked.msh", 2), ("prism_warped.msh", 1),
             ("prism_order_r0.msh", 10), ("prism_order_r1.msh", 11),
             ("prism_order_r2.msh", 12))


def scramble(src, dst, seed):
    """Write a copy of a gmsh 2.2 prism mesh with each prism's node list permuted.

    :arg src: The path of the mesh to copy.
    :arg dst: The path of the copy.
    :arg seed: The seed of the random choice. The same seed gives the same file.

    Each prism gets one of PRISM_SYMMETRIES, at random. The geometry does not
    change, but the plex cones do, so the cells present their shared facets
    in many orientations. gmsh gives the z-aligned meshes the same orientation
    on the two sides of every triangular facet, and a missing point
    permutation on those facets then gives no error. The surface elements do
    not change.
    """
    rng = random.Random(seed)
    lines = open(src).read().splitlines()
    out = []
    i = 0
    in_elements = False
    while i < len(lines):
        line = lines[i]
        if line == "$Elements":
            in_elements = True
            out += [line, lines[i + 1]]
            i += 2
            continue
        if line == "$EndElements":
            in_elements = False
        if in_elements:
            fields = line.split()
            # fields: id, type, ntags, tags..., node tags. Type 6 is the 6-node prism.
            if fields[1] == "6":
                ntags = int(fields[2])
                head, nodes = fields[:3 + ntags], fields[3 + ntags:]
                symmetry = rng.choice(PRISM_SYMMETRIES)
                line = " ".join(head + [nodes[symmetry[k]] for k in range(6)])
        out.append(line)
        i += 1
    with open(dst, "w") as fh:
        fh.write("\n".join(out) + "\n")


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


# The same prism, with the five facets as surface elements in their own
# physical groups. gmsh element type 2 is the 3-node triangle, 3 the 4-node
# quadrilateral.
REFERENCE_MARKED_MSH = """$MeshFormat
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
6
1 2 2 1 1 1 2 3
2 2 2 2 2 4 5 6
3 3 2 3 3 1 2 5 4
4 3 2 4 4 1 3 6 4
5 3 2 5 5 2 3 6 5
6 6 2 1 1 1 2 3 4 5 6
$EndElements
"""


if __name__ == "__main__":
    ref = os.path.join(HERE, "prism_reference.msh")
    with open(ref, "w") as fh:
        fh.write(REFERENCE_MSH)
    print(f"{'prism_reference.msh':<20} 1 prism (hand written, FIAT reference coordinates)")
    ref_marked = os.path.join(HERE, "prism_reference_marked.msh")
    with open(ref_marked, "w") as fh:
        fh.write(REFERENCE_MARKED_MSH)
    print(f"{'prism_reference_marked.msh':<20} 1 prism and its 5 facets (hand written)")
    for name, warp, mixed_marker in (("prism_slab.msh", False, False),
                                     ("prism_slab_mixed_marker.msh", False, True),
                                     ("prism_warped.msh", True, False)):
        path = os.path.join(HERE, name)
        types = build(path, warp, mixed_marker=mixed_marker)
        # gmsh element type 6 is the 6-node prism.
        kind = f"{types[6]} prisms (gmsh type 6), no other cell type" \
            if set(types) == {6} else f"WRONG: {types}"
        print(f"{name:<20} 3D element types: {kind}")
    for level, (divisions, nlayers) in enumerate(ORDER_LEVELS):
        name = f"prism_order_r{level}.msh"
        path = os.path.join(HERE, name)
        types = build(path, True, divisions=divisions, nlayers=nlayers)
        kind = f"{types[6]} prisms (gmsh type 6), no other cell type" \
            if set(types) == {6} else f"WRONG: {types}"
        print(f"{name:<20} 3D element types: {kind}")
    name = "prism_interior_marked.msh"
    types = build_interior_marked(os.path.join(HERE, name))
    kind = f"{types[6]} prisms (gmsh type 6), no other cell type" \
        if set(types) == {6} else f"WRONG: {types}"
    print(f"{name:<20} 3D element types: {kind}")
    for name, seed in SCRAMBLED:
        scrambled = name.replace(".msh", "_scrambled.msh")
        scramble(os.path.join(HERE, name), os.path.join(HERE, scrambled), seed)
        print(f"{scrambled:<20} a scrambled copy of {name}, seed {seed}")
    print("\nA pure-prism file matters. PETSc rewrites the cell type to")
    print("DM_POLYTOPE_TRI_PRISM_TENSOR when a file holds tetrahedra AND prisms")
    print("(see petsc/src/dm/impls/plex/plexgmsh.c:1755). That variant uses a")
    print("different cone ordering, so the permutation of plan section 5.3")
    print("would not apply.")
