# Hepatic vascular datasets

This workflow downloads upstream datasets and converts **one segmented vascular
class at a time** into a legacy ASCII `UNSTRUCTURED_GRID` containing `VTK_LINE`
cells and the arrays consumed by MetricFlowX. It complements the ADAN arterial
workflow with image-derived hepatic geometry. The solver equations are unchanged.

The output is an exploratory voxel-centerline model. Import compatibility does
not establish anatomical accuracy or physiological validity. In particular,
wall properties and terminal circuits cannot be inferred from a lumen mask.

## Sources and reuse terms

The checked download catalog is `tools/hepatic/sources.json`. New scripts and
this documentation use the project's MIT license; upstream data retain their
own terms. Downloaded data, generated networks and provenance are ignored under
`datasets/hepatic/` and are not included in this repository or release assets.

| Catalog name | Contents | Download size | Verification |
| --- | --- | --- | --- |
| `ircad01_case01` | Patient 1, including separate `artery`, `portalvein`, `venoussystem` and liver masks, CT and surface archives | About 41 MB | SHA-256 of every outer ZIP file member |
| `vsnet` | 303 reannotated MSD venous masks; original CT not included | About 9.5 MB | Commit-pinned URL and archive SHA-256 |
| `lircad` | 77 CT cases, portal/hepatic venous labels and selected anatomical subbranch labels | 13.6 GB | Version-specific Zenodo URL and provider-published MD5 |

- **IRCAD:** Soler et al. (2010), *3D image reconstruction for comparison of
  algorithm database*, IRCAD. The [provider page](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/)
  and the downloaded `LICENSE.txt` specify **CC BY-NC-ND 4.0**. The scripts
  support local adaptation; the license does not permit sharing adapted material.
  Do not publish converted IRCAD networks under the repository's MIT license.
  Patient 1 is a practical example, not a complete arterial atlas: its arterial
  mask also includes upstream vessels. Anatomical cropping is a separate task.
- **LIRCAD:** Yassine Machta and Irene Vignon-Clementel,
  [version 1.0](https://zenodo.org/records/13897086), **CC BY 4.0**;
  [Machta et al. (2024)](https://arxiv.org/abs/2411.15778).
  The main `Labels/` images use `1 = portal vein`, `2 = hepatic vein`.
  Use these images, rather than `labeled_sub_branches/`, for whole-tree extraction.
  Some cases originate from IRCAD: retain and review the underlying source terms
  as well as the LIRCAD notice before redistribution.
- **VSNet:** [Xu, Dong et al. (2025)](https://doi.org/10.1016/j.media.2025.103458),
  [author repository](https://github.com/XXYZB/VSNet). The repository has an MIT
  notice, but does not separately explain annotation licensing or the anatomical
  mapping of numeric labels 1 and 2. The original
  [MSD dataset](https://medicaldecathlon.com/) uses CC BY-SA 4.0 and requires its
  own citation. Check annotation reuse terms and anatomical label identity
  before redistribution or physiological use. `--network-kind unclassified`
  permits geometry inspection without guessing that identity.

The original MSD Task08 annotations merge vascular classes. The VSNet download
uses the authors' separate reannotations, not those original merged masks.

IRCAD's Nextcloud service generates an outer ZIP with request-dependent directory
timestamps. The downloader therefore checks the complete set of member names
and SHA-256 of their unchanged contents, rather than pinning that outer ZIP.
Its actual archive SHA-256 is still recorded in `download.json`.

## Download and convert the IRCAD example

Run commands from the repository root. Python 3.11 or newer and `curl` are needed.

```bash
python3 -m venv env
. env/bin/activate
python -m pip install -r tools/hepatic/requirements.txt
python tools/hepatic/download_sources.py ircad01_case01 --extract
```

Extraction includes the nested `MASKS_DICOM.zip`; the original CT and surface
archives are retained without automatically unpacking them. The converter reads
one DICOM mask-series directory directly. For this case foreground is **255**.

The following coefficients are **illustrative assumptions for import checks**,
not fitted patient values or recommended hepatic parameters. Resistance is
assigned equally to each terminal; it is not a calibrated equivalent whole-organ
resistance. Here `P_out=0` is a gauge-pressure reference.

```bash
python tools/hepatic/build_vtk.py \
  datasets/hepatic/ircad01_case01/3Dircadb1.1/MASKS_DICOM/artery \
  datasets/hepatic/ircad01_case01/artery.vtk \
  --label 255 --network-kind artery \
  --component largest --break-cycles --root-largest-radius \
  --young-modulus 400000 --wall-ratio 0.1 \
  --reference-pressure 10000 --terminal-resistance 1e10 --outlet-pressure 0 \
  --source-url https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/ \
  --source-license CC-BY-NC-ND-4.0

python tools/hepatic/check_conversion.py datasets/hepatic/ircad01_case01/artery.vtk
```

For the other masks, replace the input directory, output name and anatomical kind:

| Input mask directory | `--network-kind` | `--label` |
| --- | --- | --- |
| `portalvein` | `portal` | `255` |
| `venoussystem` | `hepatic` | `255` |

`--component largest` explicitly discards all but the largest 26-connected mask
component. Alternatively use its zero-based rank by descending voxel count.
The provenance reports all component sizes and the number of discarded voxels.
Without an explicit selection, a fragmented mask fails.

`--break-cycles` explicitly permits a minimum spanning tree of the skeleton graph.
Without it, remaining cycles cause failure. Inspect the reported removed edges:
the method cannot distinguish a segmentation bridge from a real connection.

`--root-largest-radius` selects the endpoint with largest distance-transform
radius; this is only a heuristic. After inspection, replace it with
`--root-point X Y Z`, giving **RAS coordinates in metres** near the desired root.
The nearest endpoint is chosen. Inspect this point and all terminals in ParaView
before assigning an inflow. Terminals include segmentation truncations and spurs.

## Download the venous collections

Each download is opt-in; requesting VSNet does not fetch LIRCAD or original MSD CT.

```bash
python tools/hepatic/download_sources.py vsnet --extract
# Larger download: 13.6 GB, plus extraction space.
python tools/hepatic/download_sources.py lircad --extract
```

Use the same converter with a NIfTI mask path and select one numeric label.
LIRCAD's main label mapping is listed above. For VSNet, inspect the anatomy first;
`reannotated/hp240.nii.gz` is a thin-slice example with approximately 0.8 mm axial
spacing. Supply `--network-kind unclassified` until the class is identified.

NIfTI coordinates follow the full image affine, including rotation, reflections
and translation. Header spatial units are converted to metres. If units are
missing, conversion fails unless an explicit `--length-unit mm`, `meter` or
`micron` is supplied. Sheared affines are rejected because the distance transform
assumes orthogonal voxel axes. DICOM slices are sorted by patient position and
checked for consistent series, spacing and orientation. DICOM LPS millimetres
are converted to RAS metres. Do not align these networks with unconverted IRCAD
VTK surfaces by assuming their coordinate conventions match. Use the corresponding
liver mask's same affine when creating the mechanical parenchyma geometry.

## Geometry and solver arrays

The selected component is cropped and padded, then skeletonized with Lee 3D
thinning. Each skeleton voxel becomes one point. Edges initially connect its 26
neighbours. A local diagonal is removed if a common neighbour supplies two
strictly shorter edges, eliminating redundant triangles at bends. Any remaining
cycle cutting requires `--break-cycles`; a deterministic minimum spanning tree
uses physical edge lengths. The rooted tree is oriented away from boundary 0.

Skeletonization is performed on the voxel lattice without isotropic resampling.
Coordinates and the Euclidean distance transform account for anisotropic spacing,
but thinning is not invariant to resolution. Thick slices, open ends, short spurs
and partial-volume effects can substantially bias geometry and radius. Some tiny
components can disappear during thinning; these fail instead of producing an
empty mesh. No extrapolation of missing distal vessels or microcirculation occurs.

Each skeleton edge is a separate solver vessel, with `vessel_id = cell index`.
This preserves MetricFlowX's property-array indexing contract and allows radius
to vary from edge to edge. It does not identify each voxel edge as a distinct
anatomical artery. There is no centerline smoothing or branch merging.

For an edge joining points `a,b`, let `r` be the arithmetic mean of their
foreground-to-background-voxel-centre distances in metres. The output uses:

| Array | Definition / unit |
| --- | --- |
| `r_d` | `r`, m |
| `a0`, `a_d` | `pi*r^2`, m² |
| `L` | Actual straight-cell length, m |
| `E` | Required `--young-modulus`, Pa |
| `h_wall` | Required `--wall-ratio` multiplied by `r`, m |
| `p_d` | Required `--reference-pressure`, Pa |
| `p0` | Zero, Pa; no additional external-pressure offset |
| `boundary_id` | Root 0, interior 255, unique other endpoint IDs skipping 255 |
| `R1`, `C` | Zero; single-resistance terminal model |
| `R2` | Required resistance **per terminal**, Pa s/m³; zero elsewhere |
| `P_out` | Required terminal reference pressure, Pa; zero elsewhere |

The companion `.provenance.json` records source hashes, source URL and license,
affine, source voxel indices, point radii, root, terminal IDs, component selection,
removed cycle edges, software versions, physical assumptions and VTK SHA-256.
`check_conversion.py` checks the output hash, required solver contract, tree
connectivity, source-affine coordinates, lengths, areas, coefficients and terminal
mapping. It does not certify segmentation quality or recover source radii anew.

A hepatic venous geometry is importable but **not a validated drainage model**.
Physiological venous drainage has distributed inflows and central outflow; the
single-root boundary assignment here is geometric. Venous collapse, sinusoidal
resistance, dual arterial/portal supply and tissue coupling need their own model
and boundary conditions. The current elastic wall law is unchanged.

## Validation

Offline tests run in the dataset CI job and need no external data:

```bash
python -m unittest discover -s tests -p test_hepatic_tools.py -v
```

The optional native import check compiles the **current checkout's** VTK reader
and verifies cell/vertex order, coordinates, property-array indices and all solver
arrays against VTK's reader. It requires deal.II with VTK support:

```bash
cmake -S tools/hepatic/import_smoke -B /tmp/hepatic_import_build \
  -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/deal.II
cmake --build /tmp/hepatic_import_build -j 2
/tmp/hepatic_import_build/hepatic_import_smoke datasets/hepatic/ircad01_case01/artery.vtk
```

The native executable initializes MPI; run it in an environment allowing MPI
initialization. It checks import, not time integration or physiological accuracy.

Local import checks on 2026-10-26 used scikit-image 0.26.0 and deal.II 9.8.0-rc1.
Fresh IRCAD/VSNet downloads passed their checksum checks. The following exports
passed both the Python consistency checker and the native C++ importer, with
explicit cycle cutting and largest-radius root selection:

| Input | Line cells | Terminal IDs other than root | Cycle edges removed |
| --- | ---: | ---: | ---: |
| IRCAD patient 1, artery | 1,249 | 67 | 35 |
| IRCAD patient 1, portal vein | 2,610 | 93 | 30 |
| IRCAD patient 1, venous system | 2,545 | 92 | 10 |
| VSNet hp240, numeric label 1 | 80 | 2 | 0 |
| VSNet hp240, numeric label 2 | 1,802 | 52 | 7 |

None of these examples discarded mask components. Counts describe the extracted
voxel graphs, not clinically verified anatomical vessel counts. The full LIRCAD
archive was not downloaded or converted in this check; its catalog entry was
checked against the provider's published file size, checksum and label mapping.
