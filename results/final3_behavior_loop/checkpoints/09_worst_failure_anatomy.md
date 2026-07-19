## Worst Failure Anatomy

- Worst-failure test isolates top 500 errors by regret, not by count.
- Worst failure pairs:
- `timsort_to_introsort`: n=253, mean regret=53.56 us, median n=13321, median margin=36.63 us.
- `heapsort_to_introsort`: n=103, mean regret=86.71 us, median n=184103, median margin=46.92 us.
- `timsort_to_heapsort`: n=75, mean regret=36.94 us, median n=8606, median margin=31.83 us.
- `introsort_to_heapsort`: n=34, mean regret=66.64 us, median n=13518, median margin=43.60 us.
- `heapsort_to_timsort`: n=22, mean regret=49.62 us, median n=2998, median margin=2.02 us.
- `introsort_to_timsort`: n=13, mean regret=47.00 us, median n=3464, median margin=0.54 us.
- Feature shift in worst failures:
- `n_elements`: worst median=1.332e+04, all-failure median=2500, all-data median=501.
- `adj_sorted_ratio`: worst median=0.5102, all-failure median=0.519, all-data median=0.5302.
- `inversion_ratio`: worst median=0.4949, all-failure median=0.4865, all-data median=0.4686.
- `runs_ratio`: worst median=0.421, all-failure median=0.623, all-data median=0.5054.
- `longest_run_ratio`: worst median=0.001862, all-failure median=0.005, all-data median=0.02.
- `duplicate_ratio`: worst median=0.07401, all-failure median=0.4184, all-data median=0.01285.
- `top5_freq_ratio`: worst median=0.003811, all-failure median=0.1594, all-data median=0.02.
- `entropy_ratio`: worst median=0.7835, all-failure median=0.6463, all-data median=0.7573.
