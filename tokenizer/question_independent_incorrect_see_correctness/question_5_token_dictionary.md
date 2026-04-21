# Question 5 Manual Token Dictionary

This file documents the manual, question-scoped token dictionaries used for `question_id = 5`.
Tokenization was assigned trace-by-trace semantically (not regex token matching).

## Scope A: Tetrahedron Question

- `TETRA_HYPOTHESIZE_EQUIFACIAL_STRUCTURE`: infer opposite-edge equality implies equifacial/isosceles tetrahedron.
- `TETRA_INSTANTIATE_FACE_EDGE_LENGTHS`: define side variables and map given edge lengths.
- `TETRA_COMPUTE_FACE_AREA`: compute area of one face correctly.
- `TETRA_MISCOMPUTE_FACE_AREA`: incorrect face-area derivation.
- `TETRA_PROPAGATE_WRONG_SURFACE_AREA`: carry wrong area into total surface area.
- `TETRA_COMPUTE_TOTAL_SURFACE_AREA`: compute total area from face area.
- `TETRA_HYPOTHESIZE_RECTANGULAR_BOX_EMBEDDING`: introduce cuboid embedding idea.
- `TETRA_SOLVE_BOX_DIMENSIONS`: solve for box dimensions from diagonal equations.
- `TETRA_COMPUTE_TETRA_VOLUME`: compute correct tetrahedron volume.
- `TETRA_MISCOMPUTE_VOLUME_AS_HALF_BOX`: incorrect V = xyz/2 assumption.
- `TETRA_MISAPPLY_CLOSED_FORM_VOLUME_FORMULA`: incorrect symbolic volume formula usage.
- `TETRA_COMPUTE_INRADIUS_FROM_V_AND_S`: apply r = 3V/S.
- `TETRA_RATIONALIZE_INRADIUS`: algebraic rationalization/simplification of r.
- `TETRA_EXTRACT_M_N_P`: read m,n,p from correct radius form.
- `TETRA_EXTRACT_M_N_P_FROM_WRONG_RADIUS`: read m,n,p from incorrect radius form.
- `TETRA_VALIDATE_NUMBER_THEORY_CONSTRAINTS`: coprime/squarefree checks.
- `TETRA_COMPUTE_FINAL_SUM`: compute m+n+p.

## Scope B: Polynomial Dialogue Question

- `POLY_INSTANTIATE_ROOT_VARIABLES_AND_VIETA`: set roots and compare coefficients via Vieta.
- `POLY_DERIVE_SQUARE_SUM_CONSTRAINT`: derive r1^2 + r2^2 + r3^2 = 81.
- `POLY_ENUMERATE_INTEGER_ROOT_TRIPLES`: enumerate valid positive integer root triples.
- `POLY_ENUMERATE_INTEGER_ROOT_TRIPLES_WITH_ERRORS`: flawed enumeration/mapping.
- `POLY_COMPUTE_CANDIDATE_A_C_PAIRS`: compute candidate (a,c) values from triples.
- `POLY_COMPUTE_WRONG_CANDIDATE_A_C_PAIRS`: compute incorrect candidate set.
- `POLY_INTERPRET_DIALOGUE_CONSTRAINT_ON_A`: apply dialogue constraints correctly.
- `POLY_MISINTERPRET_DIALOGUE_CONSTRAINT_ON_A`: wrong dialogue inference.
- `POLY_IDENTIFY_TWO_FEASIBLE_C_VALUES`: identify the correct two c values.
- `POLY_IDENTIFY_WRONG_TWO_C_VALUES`: identify wrong two c values.
- `POLY_COMPUTE_FINAL_C_SUM`: sum the two chosen c values.
