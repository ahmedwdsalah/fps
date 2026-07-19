## Size and Structure Interaction Patterns

- Interaction test checks whether one feature still matters after size changes.
- Strongest size x structure regions:
- `n_elements x inversion_ratio` size 250 to 500, feature 0 to 0.07016: timsort 99.9%, n=10,389, success=99.8%, p95 regret=0.00 us.
- `n_elements x adj_sorted_ratio` size 250 to 500, feature 0.9039 to 1: timsort 99.8%, n=10,067, success=99.6%, p95 regret=0.00 us.
- `n_elements x inversion_ratio` size 50 to 250, feature 0 to 0.07016: timsort 99.7%, n=11,514, success=99.6%, p95 regret=0.00 us.
- `n_elements x inversion_ratio` size 500 to 1000, feature 0 to 0.07016: timsort 99.7%, n=11,679, success=99.7%, p95 regret=0.00 us.
- `n_elements x inversion_ratio` size 1000 to 2513, feature 0 to 0.07016: timsort 99.6%, n=12,500, success=99.5%, p95 regret=0.00 us.
- `n_elements x adj_sorted_ratio` size 500 to 1000, feature 0.9039 to 1: timsort 99.4%, n=11,323, success=99.5%, p95 regret=0.00 us.
- `n_elements x runs_ratio` size 250 to 500, feature 5.432e-06 to 0.1791: timsort 99.3%, n=11,345, success=99.1%, p95 regret=0.00 us.
- `n_elements x adj_sorted_ratio` size 50 to 250, feature 0.9039 to 1: timsort 99.2%, n=12,007, success=99.3%, p95 regret=0.00 us.
- `n_elements x runs_ratio` size 50 to 250, feature 5.432e-06 to 0.1791: timsort 99.1%, n=12,745, success=99.1%, p95 regret=0.00 us.
- `n_elements x adj_sorted_ratio` size 1000 to 2513, feature 0.9039 to 1: timsort 99.1%, n=12,728, success=99.2%, p95 regret=0.00 us.
- `n_elements x duplicate_ratio` size 50 to 250, feature 0 to 0.002309: timsort 98.5%, n=10,956, success=98.6%, p95 regret=0.00 us.
- `n_elements x top5_freq_ratio` size 50 to 250, feature 0.032 to 0.372: timsort 98.5%, n=15,190, success=98.4%, p95 regret=0.00 us.
- `n_elements x duplicate_ratio` size 50 to 250, feature 0.002309 to 0.1202: timsort 98.4%, n=7,124, success=98.5%, p95 regret=0.00 us.
- `n_elements x duplicate_ratio` size 250 to 500, feature 0.002309 to 0.1202: timsort 98.3%, n=12,164, success=98.4%, p95 regret=0.00 us.
