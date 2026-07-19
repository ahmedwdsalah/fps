## Structural Regions Where Each Algorithm Wins

- Each structural feature was bucketed into quantile regions.
- For each region, the loop measured which of timsort/introsort/heapsort is the dominant runtime winner.
- Strongest regions found:
- `adj_sorted_ratio` 0.9039 to 0.9078: timsort dominates with 99.6% over 30,322 arrays; median winner margin 13.33 us.
- `inversion_ratio` 0.06146 to 0.07016: timsort dominates with 99.5% over 30,322 arrays; median winner margin 11.92 us.
- `runs_ratio` 0.155 to 0.1791: timsort dominates with 99.0% over 30,322 arrays; median winner margin 4.46 us.
- `runs_ratio` 0.1791 to 0.2034: timsort dominates with 98.5% over 30,322 arrays; median winner margin 8.25 us.
- `inversion_ratio` 0 to 0.06146: timsort dominates with 98.1% over 30,323 arrays; median winner margin 3.96 us.
- `adj_sorted_ratio` 0.9078 to 1: timsort dominates with 97.2% over 30,323 arrays; median winner margin 3.21 us.
- `top5_freq_ratio` 0.02 to 0.032: timsort dominates with 96.8% over 30,322 arrays; median winner margin 0.92 us.
- `top1_freq_ratio` 0.004 to 0.00431: timsort dominates with 96.3% over 30,323 arrays; median winner margin 0.75 us.
- `top5_freq_ratio` 0.01 to 0.014: timsort dominates with 95.6% over 30,322 arrays; median winner margin 1.54 us.
- `top5_freq_ratio` 0.014 to 0.02: timsort dominates with 95.5% over 30,323 arrays; median winner margin 0.87 us.
- `duplicate_ratio` 0: timsort dominates with 95.4% over 30,323 arrays; median winner margin 1.58 us.
- `longest_run_ratio` 0.12 to 1: timsort dominates with 95.4% over 30,323 arrays; median winner margin 1.79 us.
