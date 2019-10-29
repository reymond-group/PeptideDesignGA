# Peptide Design Genetic Algorithm (PDGA)
run `python run_PDGA.py population-size mutation-rate generation-gap query similarity-threshold topology`

e.g. `run_PDGA.py 50 1 0.8 Ala-Leu-Cys1-His-Gaba-Cys1-Ile 300 cyclic`
where:
- each generation will have `50` individuals
- 80% `(0.80)` of the individuals will be replaced, and 100% `(1)` of the new ones will be mutated
- PDGA will try to reach the MXFP value of the query `Ala-Leu-Cys1-His-Gaba-Cys1-Ile`
- compounds with CBD smaller than `300` from `Ala-Leu-Cys1-His-Gaba-Cys1-Ile` will be annotated in a results file
- the generated sequences will have cyclic topology.

`topology` can be `linear`, `cyclic`, or `dendritic`.

Sequences building blocks:
- the 20 natural amino acids as three-letters code (e.g. `Ala`)
- `Orn`	Ornithine
- `Hyp`	Hydroxyproline
- `bAla` Beta-alanine
- `Gaba` Gamma-aminobutyric acid
- `a5a` Delta-aminopentanoic acid 
- `a6a` Epsilon-aminohexanoic acid
- `a7a` Zeta-aminoheptanoic acid
- `a8a` Eta-aminooctanoic acid
- `a9a`	Theta-aminononaanoic acid
- `Dap` 2,3-diaminopropionic acid as branching unit
- `Dab` 2,4-diaminobutyric acid as branching unit
- `BOrn` Ornithine as branching unit
- `BLys` Lysine as branching unit
- `cy` Head-to-tail cyclization. It is always placed at the beginning (left, N terminus) of the sequence.
- `Cys1` First pair of cyclizes cysteines. Always in pair, never next to each other.
- `Cys2` Second pair of cyclizes cysteines. They are always present in pair, never next to each other, present only if Cys1 is already part of the sequence.
- `Cys3` Third pair of cyclizes cysteines. They are always present in pair, never next to each other, present only if Cys1 and Cys2 are already part of the sequence.
- `Ac` N-terminus acetylation. It is always placed at the beginning (N-terminus, left) of the sequence
- `NH2` C-terminus amide. It is always placed at the end (C-terminus, right) of the sequence

By default, all building blocks are used. Using the method `exclude_buildingblock(bb)` is possible to exclude all building blocks with the exception of the head to tail cyclization (cy) and the cyclized cysteines (Cys1, Cys2, Cys3), to exclude cyclization use “topology = linear”. 

By default, PDGA stops after CBD = 0 from the query MXFP value is found 10 times. Using the method `ga.set_time_limit('hh:mm:ss')` is possible to stop PDGA after a chosen time.

`run_PDGA.py` uses class PDGA in `PDGA_class.py`.

To run the genetic algorithm with MXFP as fitness function, a valid Chemaxon licence is required and the specified libraries needs to be downloaded.
