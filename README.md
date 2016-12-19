# HIV fitness landscape
Infer the fitness landscape of HIV-1 from longitudinal, deep sequencing intrapatient data.

## Results
- The mutation rates are in `data/mutation_rates/mutation_rate_*.tsv` in logarithms of 10, where the star covers the variation of parameters for the selection of the sites (see Fig. S2) and whether or not _env_ was included.
- The approximately neutral sites that contribute to the mutation rates are in `data/mutation_rates/mutation_rate_positions_*.txt`, in 0-based HXB2 reference coordinates. The star has the same value like above.
- The site-specific fitness costs in HXB2 coordinates are located in `data/nuc_*_selection_coefficients_any.tsv`, where `*` indicates any of several genomic regions: gag, pol, env, nef, vif, vpu, and vpr.

## Folders
- `src`: scripts
- `data`: additional data (HIVEVO_access data must be present too)


