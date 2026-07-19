## Domain-Level Behavior Patterns

- Domain split tests whether global pattern hides different behavior per source.
- Domain winner and success patterns:
- `Crypto`: n=100,000, dominant=timsort 85.1%, tim=85.1%, intro=3.7%, heap=11.1%, success=88.62%, mean regret=0.32 us.
- `Stock`: n=100,000, dominant=timsort 82.2%, tim=82.2%, intro=4.3%, heap=13.4%, success=89.57%, mean regret=0.45 us.
- `Earthquake`: n=100,003, dominant=timsort 72.6%, tim=72.6%, intro=6.6%, heap=20.9%, success=86.60%, mean regret=0.20 us.
- `Weather`: n=3,220, dominant=heapsort 39.4%, tim=35.5%, intro=25.0%, heap=39.4%, success=65.47%, mean regret=5.22 us.
