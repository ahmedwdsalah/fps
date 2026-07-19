## Class-Specific Success Indicators

- Per-class test avoids `timsort` dominance hiding intro/heap signals.
- Best class-specific indicators:
- `introsort`: `n_elements` 1.338e+04 to 1.841e+05, n=1,544, class share=10.0%, margin=2.25 us, success=85.0%.
- `introsort`: `length_norm` 0.07269 to 1, n=1,544, class share=10.0%, margin=2.25 us, success=85.0%.
- `introsort`: `duplicate_ratio` 0.9922 to 0.9999, n=1,544, class share=10.0%, margin=0.90 us, success=76.0%.
- `introsort`: `longest_run_ratio` 4.345e-05 to 0.0008703, n=1,544, class share=10.0%, margin=1.63 us, success=71.8%.
- `introsort`: `mean_abs_diff_norm` 0.0002674 to 0.01451, n=1,544, class share=10.0%, margin=0.54 us, success=68.0%.
- `heapsort`: `top5_freq_ratio` 0.8477 to 1, n=4,671, class share=10.0%, margin=0.13 us, success=95.8%.
- `heapsort`: `mad_norm` 0.02041 to 0.02041, n=4,671, class share=10.0%, margin=0.21 us, success=95.0%.
- `heapsort`: `duplicate_ratio` 0.954 to 0.9706, n=4,670, class share=10.0%, margin=0.17 us, success=94.8%.
- `heapsort`: `duplicate_ratio` 0.9706 to 0.9837, n=4,672, class share=10.0%, margin=0.21 us, success=94.6%.
- `heapsort`: `n_elements` 3000 to 3631, n=4,670, class share=10.0%, margin=0.54 us, success=93.1%.
- `timsort`: `adj_sorted_ratio` 0.9053 to 0.9095, n=24,108, class share=10.0%, margin=9.17 us, success=99.8%.
- `timsort`: `inversion_ratio` 0.05952 to 0.06598, n=24,108, class share=10.0%, margin=11.58 us, success=99.8%.
- `timsort`: `runs_ratio` 0.176 to 0.1832, n=24,108, class share=10.0%, margin=13.46 us, success=99.8%.
- `timsort`: `top5_freq_ratio` 0.02 to 0.02497, n=24,108, class share=10.0%, margin=0.79 us, success=99.4%.
- `timsort`: `n_elements` 1095 to 1482, n=24,108, class share=10.0%, margin=5.79 us, success=99.3%.
