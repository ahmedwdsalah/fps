# Extra 6 vs Final 3 Sorting Benchmark

- source file list: `/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset.csv`
- source raw arrays: `/Volumes/k/thesis_data/f1_only_1m_packed/raw_arrays.h5`
- result CSV: `/Volumes/k/thesis_data/f1_only_1m_packed/extra6_all9_retimed_labels.csv`
- rows: `1,000,008`

## Main Question: Final 3 vs Extra 6

- `final3`: `1,000,000` (100.00%)
- `extra6`: `8` (0.00%)

## Final v5 trio winners

- `heapsort`: `915,540` (91.55%)
- `timsort`: `69,631` (6.96%)
- `introsort`: `14,837` (1.48%)

## Extra 6 winners

- `counting_sort`: `864,516` (86.45%)
- `insertion_sort`: `129,134` (12.91%)
- `shell_sort`: `3,936` (0.39%)
- `comb_sort`: `1,263` (0.13%)
- `quick_sort`: `635` (0.06%)
- `merge_sort`: `524` (0.05%)

## All 9 winners

- `heapsort`: `915,537` (91.55%)
- `timsort`: `69,629` (6.96%)
- `introsort`: `14,834` (1.48%)
- `counting_sort`: `7` (0.00%)
- `insertion_sort`: `1` (0.00%)

## Timing Summary

| algorithm | mean us | median us | p95 us |
|---|---:|---:|---:|
| `timsort` | 16.418 | 13.792 | 26.208 |
| `introsort` | 9.850 | 7.875 | 16.959 |
| `heapsort` | 8.070 | 6.416 | 14.791 |
| `quick_sort` | 18183.926 | 648.583 | 43659.944 |
| `merge_sort` | 582.186 | 332.583 | 1857.000 |
| `shell_sort` | 396.625 | 225.541 | 1204.361 |
| `counting_sort` | 81.266 | 56.666 | 171.917 |
| `insertion_sort` | 3963.964 | 1416.896 | 8093.069 |
| `comb_sort` | 923.052 | 456.792 | 3031.167 |

## Thesis Use

- final 3 win: `1,000,000` / `1,000,008` (100.00%)
- extra 6 win: `8` / `1,000,008` (0.00%)
- This is direct answer: are timsort/introsort/heapsort enough, or did excluded algorithms win meaningful share?
