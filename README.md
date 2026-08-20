# B-model

This repository contains a series of numerical benchmark models (B1–B7) developed to reproduce and investigate the viscous diapirism regime described by Keller et al. (2013).

The models are based on the coupled two-phase magma dynamics formulation presented in Keller et al. (2013), with progressive development and testing of numerical treatments including boundary conditions, porosity evolution, marker-to-node property transfer, air mixing, and source-region treatment.

## Version History

| Version | Main development                                 |
| ------- | ------------------------------------------------ |
| **B1**  | Implementation and treatment of the top boundary |
| **B2**  | Porosity update                                  |
| **B3**  | Combined top-boundary and porosity treatment     |
| **B4**  | Marker-to-node material-property interpolation   |
| **B5**  | Air-mixing treatment                             |
| **B6**  | Combined marker-to-node treatment and air mixing |
| **B7**  | Pressure-temperature aligned source mask         |

The corresponding Git tags `B1`–`B7` preserve each benchmark-development stage and can be used to inspect or compare individual versions.

## Repository Structure

```text
B-model/
├── simulation.py    # Main numerical model
├── .gitignore       # Excludes generated/output files
└── README.md        # Project description
```

The current `main` branch contains the latest version of the model.

## Numerical Output

Large numerical simulation outputs are **not stored in this repository**.

Only the source code and files required to reproduce or develop the numerical model are tracked with Git. Generated output directories are excluded to keep the repository lightweight and to maintain a clear source-code history.

## Running the Model

The current version can be run with:

```bash
python simulation.py
```

Depending on the local environment, additional Python packages and numerical dependencies may be required.

## Comparing Versions

The development history can be inspected using the Git tags:

```text
B1 → B2 → B3 → B4 → B5 → B6 → B7
```

For example, the changes between B6 and B7 can be inspected locally with:

```bash
git diff B6 B7 -- simulation.py
```

The same versions can also be inspected through the **Tags** and **Commits** pages on GitHub.

## Reference

This benchmark is based on the viscous diapirism benchmark described by **Keller et al.**

Keller, T., May, D. A., & Kaus, B. J. P. (2013).
Numerical modelling of magma dynamics coupled to tectonic deformation of lithosphere and crust.
Geophysical Journal International, 195(3), 1406–1442.
DOI: 10.1093/gji/ggt306
