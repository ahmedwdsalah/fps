# Final-Three Sorting Behavior Knowledge Report

Generated: 2026-07-13T18:37:01

## Baseline Runtime Landscape

- Rows analyzed: 303,223.
- SBS algorithm: `heapsort`.
- VBS total: 6.623571s; SBS total: 8.495169s; VBS/SBS gap: 22.03%.
- Winner share: introsort 5.09%, heapsort 15.40%, timsort 79.51%
- Meaning: this is the raw runtime landscape before explaining success or failure.

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

## Why Correct Predictions Are Correct

- Successful predictions: 266,891 (88.02%).
- Highest success indicators are regions where the true fastest class is structurally clear and the runtime margin is usually not tiny.
- `adj_sorted_ratio` 0.9048 to 0.9088: correct timsort region, 99.7% dominance, n=26,689, median margin=11.83 us.
- `top1_freq_ratio` 0.004 to 0.004024: correct timsort region, 99.7% dominance, n=26,689, median margin=0.75 us.
- `inversion_ratio` 0.06046 to 0.06737: correct timsort region, 99.7% dominance, n=26,689, median margin=12.62 us.
- `runs_ratio` 0.1779 to 0.1873: correct timsort region, 99.6% dominance, n=26,689, median margin=11.00 us.
- `runs_ratio` 0.15 to 0.1779: correct timsort region, 99.0% dominance, n=26,689, median margin=4.29 us.
- `top5_freq_ratio` 0.02 to 0.02762: correct timsort region, 99.0% dominance, n=26,689, median margin=0.87 us.
- `inversion_ratio` 0 to 0.06046: correct timsort region, 98.8% dominance, n=26,690, median margin=3.71 us.
- `top5_freq_ratio` 0.014 to 0.02: correct timsort region, 98.6% dominance, n=26,689, median margin=0.96 us.
- `adj_sorted_ratio` 0.9088 to 1: correct timsort region, 98.2% dominance, n=26,689, median margin=2.83 us.
- `top1_freq_ratio` 0.001594 to 0.002: correct timsort region, 97.9% dominance, n=26,689, median margin=2.21 us.
- `duplicate_ratio` 0: correct timsort region, 97.8% dominance, n=26,690, median margin=1.63 us.
- `top1_freq_ratio` 0.004024 to 0.007968: correct timsort region, 97.5% dominance, n=26,689, median margin=1.25 us.

## Why Failed Predictions Fail

- Failed predictions: 36,332 (11.98%).
- Low-regret failures <=1 us: 19,806; high-regret failures >1 us: 16,526.
- Main failed class-pairs:
- `timsort_to_heapsort`: n=13,482, mean regret=3.16 us, p95=9.04 us.
- `introsort_to_heapsort`: n=8,227, mean regret=0.94 us, p95=3.42 us.
- `heapsort_to_introsort`: n=4,745, mean regret=3.84 us, p95=11.00 us.
- `timsort_to_introsort`: n=4,721, mean regret=8.15 us, p95=27.71 us.
- `heapsort_to_timsort`: n=3,378, mean regret=1.28 us, p95=4.71 us.
- `introsort_to_timsort`: n=1,779, mean regret=1.49 us, p95=6.02 us.
- Strong high-regret feature regions:
- `longest_run_ratio` 0.002333 to 0.0024: timsort_to_heapsort dominates failures with 80.2%, n=1,653.
- `n_elements` 2999 to 3000: timsort_to_heapsort dominates failures with 78.8%, n=1,653.
- `length_norm` 0.01629 to 0.0163: timsort_to_heapsort dominates failures with 78.8%, n=1,653.
- `dispersion_ratio` 0.05035 to 0.06621: timsort_to_heapsort dominates failures with 73.7%, n=1,653.
- `mad_norm` 0.02144 to 0.02868: timsort_to_heapsort dominates failures with 71.1%, n=1,652.
- `longest_run_ratio` 0.002001 to 0.002333: timsort_to_heapsort dominates failures with 70.6%, n=1,652.
- `n_elements` 3000 to 3483: timsort_to_heapsort dominates failures with 68.0%, n=1,652.
- `length_norm` 0.0163 to 0.01892: timsort_to_heapsort dominates failures with 68.0%, n=1,652.
- `mean_abs_diff_norm` 0.04876 to 0.06389: timsort_to_heapsort dominates failures with 67.9%, n=1,652.
- `longest_run_ratio` 0.002786 to 0.002807: timsort_to_heapsort dominates failures with 67.3%, n=1,653.
- `iqr_norm` 0.05024 to 0.06591: timsort_to_heapsort dominates failures with 66.8%, n=1,652.
- `n_elements` 2500 to 2513: timsort_to_heapsort dominates failures with 66.5%, n=1,653.

## Representative Examples

- Representative cases exported to `tables/representative_cases.csv`.
- `Weather` n=52,585: true=introsort, pred=introsort, regret=0.00 us, margin=496.96 us, file=`weather_Singapore_2015_2020_wind_rolling_24h_std_REV.csv`.
- `Crypto` n=17,361: true=heapsort, pred=heapsort, regret=0.00 us, margin=361.92 us, file=`crypto_LTC-USD_2y_1h_rolling_5d_std_SHUF.csv`.
- `Weather` n=184,104: true=timsort, pred=timsort, regret=0.00 us, margin=4380.29 us, file=`weather_Singapore_2000_2020_precip_cumsum.csv`.
- `Earthquake` n=63: true=introsort, pred=timsort, regret=0.00 us, margin=0.00 us, file=`quake_japan_global_M5plus_20200101_20201231_latitude_REV.csv`.
- `Weather` n=184,081: true=timsort, pred=introsort, regret=659.25 us, margin=659.25 us, file=`weather_Havana_2000_2020_temp_rolling_24h_mean_PSORT10.csv`.

## Domain-Level Behavior Patterns

- Domain split tests whether global pattern hides different behavior per source.
- Domain winner and success patterns:
- `Crypto`: n=100,000, dominant=timsort 85.1%, tim=85.1%, intro=3.7%, heap=11.1%, success=88.62%, mean regret=0.32 us.
- `Stock`: n=100,000, dominant=timsort 82.2%, tim=82.2%, intro=4.3%, heap=13.4%, success=89.57%, mean regret=0.45 us.
- `Earthquake`: n=100,003, dominant=timsort 72.6%, tim=72.6%, intro=6.6%, heap=20.9%, success=86.60%, mean regret=0.20 us.
- `Weather`: n=3,220, dominant=heapsort 39.4%, tim=35.5%, intro=25.0%, heap=39.4%, success=65.47%, mean regret=5.22 us.

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

## Final Synthesis and Next Tests

- The loop builds a behavior map for the final three algorithms only.
- The main question is which structural indicators make timsort, introsort, or heapsort win.
- Success analysis explains where the model sees the same structure as the timing oracle.
- Failure analysis separates harmless boundary mistakes from expensive mistakes.
- Prediction status: predictions generated from models/xgboost_v5/xgb_v5.json.
- Next tests should target the strongest high-regret regions with controlled synthetic probes and optional fresh retiming.
