# Data in this folder

- `clean_spanner_dataset.csv` — tracked in Git (final file, small).
- `spanner_dataset_pro.csv` — deliberately excluded from Git (it is large). Regenerate it with
  `python data_generator.py` from the repository root; the benchmark scripts read it from here.
- `final_scientific_spanner.csv` and `optimized_spanner_edges.csv` — large intermediate outputs of
  earlier experiments, also excluded from Git; no current script reads them.
