# Complete ADAN arterial network: sources and reconstruction

This workflow builds a MetricFlowX input from the **arterial component of
ADAVN 2023**, an updated ADAN network. It includes every arterial row in the
published supplement: **4,041 source segments**, with 2,184 distinct artery
names, one inlet and 1,471 terminals. Each source segment becomes two straight
cells, giving 8,082 cells and 7,745 points. These counts describe the supplied
table, rather than identifying it with the unchanged ADAN 2015 dataset.

**The default uses only the published CSV and produces a nonanatomical metric
graph.** The optional anatomical reconstruction uses separately licensed
HeMoLab surfaces and requires the relevant written permission. No downloaded
surfaces, surface-derived meshes or figures are distributed in this repository.

For **arteries and veins together**, use the [full ADAVN tutorial](../tutorials/05_adavn_full/README.md),
which adds venous walls and coupled arterial–venous terminal connections.
The arterial-only workflow below remains available.

## Sources, attribution and licenses

| Material | Primary source | License and use in this workflow |
| --- | --- | --- |
| ADAVN article, equations and supplementary table | Müller, Watanabe, Toro, Feijóo and Blanco (2023), [Frontiers in Physiology 14:1162391](https://doi.org/10.3389/fphys.2023.1162391); [publisher page and copyright notice](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1162391/full) | Copyright © 2023 the authors; the publisher identifies the work as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain attribution, a license link and an indication of changes. The CSV is downloaded from the article's supplement; its header contains no separate license notice. |
| Connectivity, lengths, radii, total terminal resistance and compliance | [Data Sheet 1.CSV, direct publisher download](https://public-pages-files-2025.frontiersin.org/articles/1162391/file/Data_Sheet_1.CSV/1162391_supplementary-materials_datasheets_1_csv/1) | Covered here by the publisher's CC BY notice for the study; pinned by SHA-256. No source data are relabeled as MIT. |
| Additional model parameters and compartment definitions | [Data Sheet 2.pdf](https://public-pages-files-2025.frontiersin.org/articles/1162391/file/Data_Sheet_2.pdf/1162391_supplementary-materials_datasheets_2_pdf/2) | Supplement to the same article; reference for the original closed-loop model, not an additional input required by the default converter. |
| Optional arterial surface atlas | [HeMoLab ADAN-WEB](http://hemolab.lncc.br/adan-web/), its `ADAN.json` artery catalogue and `MESHES/arterial/SW_<artery>.vtk` files | **Separate restrictive terms, not CC BY or MIT:** [HeMoLab Terms of Use](http://hemolab.lncc.br/adan-web/doc/Terms.html). Copyright © 2018 HeMoLab-LNCC, all rights reserved. The stated default is personal, noncommercial and nonclinical use; modification, derivative works and redistribution require written permission unless otherwise specified. Check the current terms and your permission before using this mode. |
| Reconstruction programs, tests and this documentation | This repository, `tools/adan/`, `tests/test_adan_complete.py` | [Repository MIT license](https://github.com/luca-heltai/metric-flow-x/blob/main/LICENSE.md). This license covers the new code and documentation, not third-party data or atlas-derived output. |

The CSV is archived **outside Git**, with its attribution and checksums, as
[GitHub release assets](https://github.com/luca-heltai/metric-flow-x/releases/tag/datasets-final-version-v1).
The downloader uses the archived
[`adan_2023_source_v1.csv`](https://github.com/luca-heltai/metric-flow-x/releases/download/datasets-final-version-v1/adan_2023_source_v1.csv),
which is byte-for-byte identical to the publisher supplement. The
[attribution notice](https://github.com/luca-heltai/metric-flow-x/releases/download/datasets-final-version-v1/adan_2023_source_v1_attribution.txt)
and [checksum file](https://github.com/luca-heltai/metric-flow-x/releases/download/datasets-final-version-v1/adan_2023_source_v1.sha256)
are companion assets. No CSV or large VTK is committed.

The source URLs, CSV digest and individual surface digests are recorded in
[`tools/adan/sources.json`](../tools/adan/sources.json).
The snapshot was collected on **26 October 2026**. The atlas was reachable
through HTTP; HTTPS did not respond. Atlas downloads are checked against the
pinned digests even though the server uses HTTP. A changed or corrupt file
fails verification instead of being accepted silently.

The historical ADAN references are:

- Blanco, Watanabe, Passos, Lemos and Feijóo (2015), *An Anatomically Detailed
  Arterial Network Model for One-Dimensional Computational Hemodynamics*,
  IEEE TBME 62(2), 736–753,
  [doi:10.1109/TBME.2014.2364522](https://doi.org/10.1109/TBME.2014.2364522).
- Blanco, Watanabe, Dari, Passos and Feijóo (2014), *Blood flow distribution
  in an anatomically detailed arterial network model: criteria and algorithms*,
  Biomechanics and Modeling in Mechanobiology 13, 1303–1330,
  [doi:10.1007/s10237-014-0574-8](https://doi.org/10.1007/s10237-014-0574-8).

The workflow does not download those historical articles or assume their
licenses apply to the atlas. It uses the 2023 supplementary table.

A suitable attribution for the CSV-only derived data is:

> Based on Müller, Watanabe, Toro, Feijóo and Blanco (2023),
> doi:10.3389/fphys.2023.1162391, CC BY 4.0. Changes: arterial extraction,
> SI conversion, nonanatomical metric embedding, two-cell subdivision and
> adapted wall and terminal parameters. No author endorsement is implied.

This attribution and the data license are also written into the generated
provenance file. For anatomical output, retain the HeMoLab attribution and
the additional conditions of the permission you obtained; a CC BY notice on
the CSV does not authorize redistribution of surface-derived geometry.

## Download and rebuild

Run from the repository root with Python 3.10 or newer. The converter and
checker use only the Python standard library; no VTK Python package is needed.

```sh
python3 tools/adan/download_sources.py
python3 tools/adan/recover_geometry.py
python3 tools/adan/build_vtk.py
python3 tools/adan/check_conversion.py
```

Outputs are placed in `build/adan/`, which is ignored by Git. All commands accept
`--data-dir PATH`; pass the same directory to each command. The downloader
verifies cached files as well as new downloads. On systems with a custom TLS
trust store, configure Python's normal `SSL_CERT_FILE` setting to the trusted
CA bundle; do not disable certificate verification.

Generated files include:

- `adan_complete.vtk`: legacy ASCII `UNSTRUCTURED_GRID` with `VTK_LINE` cells.
- `adan_complete.mapping.json`: cell IDs, source names, source node IDs and lengths.
- `adan_complete.terminals.csv`: adapted RCR values, original vascular territories
  and original compartment connection IDs.
- `adan_complete.provenance.json`: source and output digests, geometry mode,
  coefficients, conventions, attribution and limitations.
- `download_receipt.json`, `geometry_report.json`, `conversion_checks.json`:
  source verification, reconstruction diagnostics and independent checks.

The generated VTK is not a published anatomical centerline dataset. The default
coordinates are abstract; their purpose is to represent the original metric
graph in a format the solver can import.

### Optional anatomical reconstruction

**Use this mode only with the written permission required for your intended
HeMoLab use.** `--atlas-permission` records an explicit choice to use this
restricted source; it does not grant rights or obtain permission for you.

```sh
python3 tools/adan/download_sources.py --include-atlas --atlas-permission
python3 tools/adan/recover_geometry.py --mode anatomical
python3 tools/adan/build_vtk.py
python3 tools/adan/check_conversion.py
```

The optional downloader fetches all 2,184 surfaces listed in the pinned
manifest. Existing authorized files may instead be supplied under
`build/adan/source/surfaces/`. The source CSV is still required. The scripts
never execute downloaded content. Publishing the resulting anatomical VTK or
an image derived from those surfaces is subject to HeMoLab's separate terms
and the permission obtained, not the code's MIT license.

## Graph and metric reconstruction

Let the arterial CSV define a connected graph $G=(V,E)$, with length $L_e>0$
and inlet/outlet radii for each segment. Nodes are joined by their **CSV node
IDs**, not by proximity. Anastomoses and cycles are retained. The full graph
has $|V|=3704$, $|E|=4041$, and cycle rank $|E|-|V|+1=338$.
The inlet is the unique degree-one node whose incident row has no terminal
resistance (source node 778). All other degree-one nodes are outlets; their
CSV orientation need not point toward the outlet.

For the CSV-only embedding, let $D_v$ be the shortest-path distance from the
inlet, $eta=0.01\min_e L_e$, and $\theta_v=2.399963229728653\,v$. Set

$$
\boldsymbol{x}_v=(0.9D_v,\eta\sin\theta_v,\eta\cos\theta_v).
$$

Shortest-path distances satisfy $|D_u-D_v|\le L_e$ on every edge. The bounded
transverse offsets therefore give a chord shorter than $L_e$; the coordinates
carry no anatomical interpretation.

For a segment with endpoint chord length $d_e\le L_e$, choose a unit vector
$\boldsymbol{n}_e$ perpendicular to that chord and insert

$$
\boldsymbol{m}_e=\frac{\boldsymbol{x}_u+\boldsymbol{x}_v}{2}
 +\frac{\sqrt{L_e^2-d_e^2}}{2}\boldsymbol{n}_e.
$$

Both resulting straight cells have length $L_e/2$. Thus subdivision preserves
the source metric, topology and cycle rank, without depending on a visual
layout. Linear radius interpolation preserves the source endpoint radii and
sets the midpoint radius to their arithmetic mean.

In anatomical mode, centers of circular 12-vertex rings are extracted from
the surface tubes. Ring regularity is checked; orientation is resolved using
shared source nodes and a local least-squares consistency procedure. Internal
CSV nodes are sampled at normalized arclength fractions along the recovered
polylines. Shared-node estimates are averaged. An iterative chord constraint
projection then enforces $d_e\le L_e$, and the same midpoint formula restores
each published length. The perpendicular direction follows the recovered
curve where possible. This is an approximate reconstruction, not recovery of
the original splines or a proof of globally optimal orientation alignment.

The source atlas and the table have different sampling and sometimes different
lengths. In the local full anatomical reconstruction, shared-node estimates
differed by up to approximately 12.16 mm before averaging; the subsequent
chord projection moved averaged nodes by at most approximately 3.087 mm.
Those are geometry diagnostics, not anatomical accuracy guarantees. Inspect
the generated report for the actual run. Do not treat the reconstructed
curvature as an original anatomical datum for 3D or geometric coupling.

## SI conversion and VTK contract

Lengths and radii in the supplement are in centimeters; terminal R/C are
interpreted in the CGS convention of the original model:

$$
L_{\rm SI}=10^{-2}L_{\rm cm},\quad
R_{\rm SI}=10^5R_{\rm CGS},\quad
C_{\rm SI}=10^{-5}C_{\rm CGS}.
$$

Pressure is in Pa; resistance in Pa s/m$^3$; compliance in m$^3$/Pa.
Each straight cell gets a unique, contiguous `vessel_id` equal to its cell
index. This follows the current solver's material-array indexing and keeps
taper interpolation on straight segments. `source_segment_id` and the mapping
file retain the original segment identity.

Required cell fields are `vessel_id`, `a0`, `a_d`, `E`, `h_wall`, `p_d`, `p0`,
`L`, `r_d`; `r_in` and `r_out` preserve taper. Here `L` is the **cell** length.
Point fields are `boundary_id`, `R1`, `R2`, `C`, `P_out`. The inlet ID is 0,
interior ID 255 is reserved, and distinct positive terminal IDs skip 255.

## Arterial wall law and the current elastic approximation

Equation (3) of Müller et al. (2023) gives the arterial transmural pressure

$$
p_{\rm tm}=p_{\rm ref}
 +\frac{\pi R_{\rm ref}h_{\rm ref}}{A}
   \left[E_e\epsilon+E_c\epsilon_r\log(1+e^\chi)\right]
 +\frac{K_m}{2\sqrt{AA_{\rm ref}}}\partial_t A,
$$

where $A_{\rm ref}=\pi R_{\rm ref}^2$,
$\epsilon=\sqrt{A/A_{\rm ref}}-1$ and
$\chi=(\epsilon-\epsilon_0)/\epsilon_r$.
The reference transmural pressure is $p_{\rm ref}=10^4$ Pa,
$\epsilon_0=0.25$, and $\epsilon_r=0.05$.

The mixture coefficients are from sections 2.4.1–2.4.2 and Tables 1–2:

| Reference radius [cm] | $W_e$ | $W_c$ | $W_m$ |
| --- | ---: | ---: | ---: |
| $R>0.18$ | 0.85 | 0.05 | 0.10 |
| $0.07\le R\le0.18$ | 0.65 | 0.20 | 0.15 |
| $R<0.07$ | 0.45 | 0 | 0.55 |

$$
E_e=W_e(4\times10^5\ {\rm Pa}),\quad
E_c=W_c(10^8\ {\rm Pa}),\quad
K_m=W_m(3.6\times10^4\ {\rm Pa\,s}).
$$

Equation (43) specifies wall thickness, with $R_{\rm cm}$ in centimeters:

$$
\frac{h_{\rm ref}}{R_{\rm ref}}
 =0.2802e^{-5.053R_{\rm cm}}+0.1324e^{-0.1114R_{\rm cm}}.
$$

The extra VTK fields `adavn_E_e`, `adavn_E_c`, `adavn_K_m`,
`adavn_epsilon_0`, `adavn_epsilon_r`, `adavn_p_ref` preserve these coefficients
for future solver work. They are evaluated at each straight cell's mean
reference radius; mixture categories may change along an original segment.
The endpoint radii allow future pointwise reconstruction of these quantities.
**These fields do not activate the nonlinear or viscoelastic law.** Integrating
the $\partial_t A$ term requires corresponding residual/Jacobian changes and
is outside this data-workflow change.

For the existing square-root law, write
$F(\epsilon)=E_e\epsilon+E_c\epsilon_r
\log(1+e^{(\epsilon-\epsilon_0)/\epsilon_r})$ and
$F_0=F(0)$, $F'_0=E_e+E_c/(1+e^{\epsilon_0/\epsilon_r})$.
The converter sets

$$
E_{\rm eq}=\frac32\left(\frac12F'_0-F_0\right),\qquad
p_d=p_{\rm ref}+\frac{h_{\rm ref}}{R_{\rm ref}}F_0,
$$

with `p0=0`, `a0=a_d=A_ref`. Substitution in

$$
p_{\rm current}(A)=p_d+
\frac{4\sqrt\pi E_{\rm eq}h_{\rm ref}}{3A_{\rm ref}}
(\sqrt A-\sqrt{A_{\rm ref}})
$$

matches the value and $\partial p/\partial A$ of the **elastic part** of the
ADAVN law at $A=A_{\rm ref}$. The small collagen activation already present
at zero strain is retained. This is a local tangent approximation; it does
not reproduce the full nonlinear response or viscous dissipation.

## Future venous wall law

The converter above selects **arterial rows only**. For a future venous
extension, ADAVN equation (6) provides a collapsible tube law. With
$\alpha=A/A_{\rm ref}$,

$$
p_{\rm tm}^{\rm ven}=p_{\rm ref}^{\rm ven}
 +K(\alpha^m-\alpha^n)
 +\frac{K_m^{\rm ven}}{A_{\rm ref}\sqrt A}\partial_t A.
$$

Section 2.4.2 uses **$m=10$, $n=-3/2$ for ordinary veins**, and
**$m=1/2$, $n=0$ for dural sinuses**. The negative exponent provides a
collapse response at negative transmural pressure; the positive exponent
provides distension stiffening. The reference pressure is
$p_{\rm ref}^{\rm ven}=666.666$ Pa (the paper's 6666.66 dyn/cm$^2$).
The stiffness follows from a chosen reference wave speed:

$$
K=\frac{\rho c_0^2}{m-n},\qquad
c^2(A)=\frac K\rho\left(m\alpha^m-n\alpha^n\right).
$$

An elastic implementation can start with the first two pressure terms.
The paper additionally specifies a radius-dependent $c_0$ in equation (44)
and $K_m^{\rm ven}=\widehat K R_{\rm ref}$ with
$\widehat K=70.898$ Pa s, so $K_m^{\rm ven}$ has units Pa s m in SI.
Do not substitute the arterial viscoelastic coefficient: the denominators
and coefficient dimensions differ between equations (3) and (6).

Both nonlinear arterial and venous implementations will require consistent
pressure derivatives, wave-speed derivatives, Riemann compatibility and
junction/boundary equations. Adding only a pressure evaluator to the current
square-root-law formulation would not complete that implementation. The
viscoelastic terms also require the corresponding time-derivative residual
and Jacobian contributions. This PR supplies data and documentation, without
changing those solver equations.

## Terminal adaptation and inlet data

Original total terminal resistance $R_T$ and compliance $C$ are retained.
For $C>0$, the added RCR split is

$$
c_{\rm ref}=\sqrt{\frac{2E_{\rm eq}h_{\rm ref}}{3\rho R_{\rm ref}}},\quad
R_1=\frac{\rho c_{\rm ref}}{\pi R_{\rm ref}^2},\quad R_2=R_T-R_1.
$$

For $C=0$, use a single resistance: $R_1=0$, $R_2=R_T$. A nonpositive
$R_2$ is rejected, not clipped. The default density is 1040 kg/m$^3$;
`build_vtk.py --density VALUE` must agree with the solver configuration.

The default `P_out=0 Pa`, configurable by `--p-out VALUE`, is an added
fixed-pressure closure. The source venous, heart, coronary, cerebrospinal-fluid
and microcirculatory coupling is **not reproduced**. Original territory and
connection IDs remain in the terminal table for later reconstruction.
No arbitrary inflow waveform is supplied as if it were published ADAVN data.
Choose the inflow separately for the intended experiment.

Point the case parameter `Vtk file path for mesh input` to
`build/adan/adan_complete.vtk`, select `Outlet boundary condition type = RCR`,
and use the same density. Start with zero global refinements when checking
this already subdivided network. See [Configuration](configuration.md) for
the remaining case parameters.

## Verification and limits

`check_conversion.py` compares all original node connections, endpoint radii,
segment lengths, and terminal R/C values, and independently compares the active
elastic tangent with a finite-difference derivative of the published equation.
It also invokes the repository's VTK input validator.
`tests/test_adan_complete.py` uses offline fixtures for reversed terminal
orientation, zero-compliance outlets, radius-category endpoints, metric
subdivision, checksum rejection and the optional-atlas permission guard.

Checks establish an importable input and the stated conversion invariants.
They do not establish convergence, conservation during time integration,
physiological validation, equivalence to the full closed-loop ADAVN model,
or the anatomical accuracy of the recovered geometry.
