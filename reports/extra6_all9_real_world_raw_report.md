# Extra 6 vs Final 3 Sorting Benchmark - Real-World Raw

- source index: `/Volumes/k/thesis_data/real_world_10k/index.csv`
- source raw dir: `/Volumes/k/thesis_data/real_world_10k/raw`
- result CSV: `/Volumes/k/thesis_data/real_world_10k/extra6_all9_retimed_labels.csv`
- rows: `303,223`

## Main Question: Final 3 vs Extra 6

- `final3`: `262,225` (86.48%)
- `extra6`: `40,998` (13.52%)

## Final v5 trio winners

- `heapsort`: `235,062` (77.52%)
- `timsort`: `36,563` (12.06%)
- `introsort`: `31,598` (10.42%)

## Extra 6 winners

- `comb_sort`: `108,113` (35.65%)
- `counting_sort`: `88,346` (29.14%)
- `insertion_sort`: `84,902` (28.00%)
- `quick_sort`: `11,532` (3.80%)
- `shell_sort`: `7,921` (2.61%)
- `merge_sort`: `2,409` (0.79%)

## All 9 winners

- `heapsort`: `207,507` (68.43%)
- `insertion_sort`: `37,548` (12.38%)
- `introsort`: `28,356` (9.35%)
- `timsort`: `26,362` (8.69%)
- `comb_sort`: `2,421` (0.80%)
- `shell_sort`: `821` (0.27%)
- `counting_sort`: `200` (0.07%)
- `quick_sort`: `7` (0.00%)
- `merge_sort`: `1` (0.00%)

## Timing Summary

| algorithm | mean us | median us | p95 us |
|---|---:|---:|---:|
| `timsort` | 58.211 | 17.375 | 187.625 |
| `introsort` | 34.230 | 9.334 | 103.209 |
| `heapsort` | 33.469 | 8.291 | 102.084 |
| `quick_sort` | 3277.565 | 24.000 | 705.537 |
| `merge_sort` | 185.925 | 23.250 | 433.371 |
| `shell_sort` | 184.526 | 22.083 | 426.367 |
| `counting_sort` | 65.793 | 26.917 | 188.121 |
| `insertion_sort` | 10324.500 | 22.625 | 3284.354 |
| `comb_sort` | 244.012 | 16.958 | 378.329 |

## Thesis Use

- final 3 win: `262,225` / `303,223` (86.48%)
- extra 6 win: `40,998` / `303,223` (13.52%)
