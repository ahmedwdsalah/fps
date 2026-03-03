# Related Work Analysis — Reading Notes from Source Documents

> **Date:** 2025-06-XX (updated after full second re-read of all 4 sources)  
> **Sources read directly from:** `online_blogs/` folder  
> **Files:** `blog_1.md`, `blog_2.md`, `Adaptive Hybrid Sort by Balasubramanian.pdf`, `NeurIPS 2023 Sorting with Predictions.pdf`  
> **Verification:** Every claim below was verified against the original source text. PDFs were fully extracted (11pp + 22pp) and read page-by-page.

---

## 1. Balasubramanian — "Adaptive Hybrid Sort (AHS)"

**Source:** SSRN preprint (NOT peer-reviewed — watermark on every page: "This preprint research paper has not been peer reviewed")  
**File:** `online_blogs/Adaptive Hybrid Sort by Balasubramanian.pdf` (11 pages + references)  
**Blog summary:** `online_blogs/blog_1.md`

### What They Do

- Uses XGBoost to select between **Counting Sort, Radix Sort, QuickSort** (+ Insertion Sort for n ≤ 20)
- Decision based on a state vector **v = (n, k, H)** — only **3 features**:
  - `n` = array size (cardinality)
  - `k` = key range (max − min + 1)
  - `H` = Shannon entropy: −Σ (fᵢ/n) log₂(fᵢ/n)
- Hierarchical Finite State Machine + XGBoost Classifier:
  - If n ≤ 20 → Insertion Sort
  - If k ≤ 1000 → Counting Sort
  - If k > 10⁶ and H < 0.7·log₂(k) → Radix Sort
  - Else → QuickSort (median-of-three pivot)
- XGBoost trained on **10,000 synthetic arrays**:
  - n sampled uniformly from [10³, 10⁶]
  - k sampled uniformly from [10², 10⁶]
  - Distributions: Uniform, Gaussian (μ=0, σ=k/4), Zipfian (skew=1.5)
- Model: 92.4% accuracy, F1=0.89, 0.2ms inference latency, 45s training
- 8-bit quantized model reduces size from 4MB to 1MB
- ML vs Rules comparison: ML 92.4% accuracy vs rule-based 84.6%

### Threshold Calibration

- Multi-objective Bayesian optimization: min[α·T(nₜ,kₜ) + (1−α)·M(nₜ,kₜ)], α=0.7
- Grid search: nₜ ∈ [10, 50], kₜ ∈ [500, 5000], then 100 iterations Bayesian optimization
- Hardware-aware: kₘₐₓ = L3_Cache / (4 × Thread_Count)
- Final thresholds: nₜ=20, kₜ=1024, kₘₐₓ=10⁶

### Real-World Datasets Used

- NYC Taxi timestamps: n=10⁷, k=10⁹, H=8.2
- IoT sensor readings: n=10⁶, k=500, H=1.1
- Genomic k-mers: n=10⁸, k=4³⁰, H=3.7

### Performance Claims

| Dataset Size | AHS Time | TimSort Time | Memory |
|---|---|---|---|
| 10⁶ | 0.21s | 0.38s | 0.8GB |
| 10⁷ | 2.1s | 3.8s | 8.0GB |
| 10⁹ | 210s | 380s | 8.0GB |

- Claims 40-45% speedup over TimSort at large scale
- Hardware acceleration: OpenCL GPU (3.5× on RTX 3080), AVX2/NEON SIMD
- Parallel speedup factors: Radix Sort 1.79×, QuickSort 1.12×, Counting Sort 0.95× (negligible for Counting)
- Implementation: TypeScript + C++
- Benchmarked on Windows 11/WSL2, Intel i7 + RTX 3080

### Acknowledged Limitations

1. Entropy calculation overhead (~0.2ms) significant for ultra-low-latency systems
2. **Integer-only** — no floats or strings
3. ML misclassification risk (~7.6% error rate), proposes RL as future work
4. Hardware-specific optimizations are platform-dependent

### Critical Assessment

- **Not peer-reviewed** — SSRN preprint
- Only 3 features is extremely coarse — cannot detect presortedness, runs, inversions, distribution shape
- Hard-coded thresholds (k ≤ 1000, k > 10⁶) make XGBoost partially redundant — the FSM already decides most cases before ML is consulted
- Claims of 210s for 10⁹ elements vs TimSort's 380s are extraordinary and unverified by independent benchmarks
- Algorithm portfolio (Counting, Radix, Quick) is specialized for integer distributions — not general-purpose
- Future work mentions RL for continuous learning — exactly what our bandit does

---

## 2. DeepMind — "AlphaDev" (Nature 2023)

**Source:** Nature, 618(7964):257–263, 2023 (Mankowitz et al.)  
**File:** `online_blogs/blog_2.md`

### What They Do

- Used **AlphaZero-style reinforcement learning** to discover faster sorting algorithm **implementations** at the assembly instruction level
- Modeled sorting as a **single-player game**:
  - State: current program (sequence of assembly instructions) + CPU register/memory state
  - Action: choose next assembly instruction (mov, cmp, cmov, jmp, etc.)
  - Reward: correctness (sorts all inputs?) + latency (fewer instructions = higher reward)
- Discovered novel **"swap and copy moves"** — instruction sequences humans hadn't found
- Applied to fixed-length sort networks: sort3, sort4, sort5

### Key Results

| Target | Improvement |
|---|---|
| sort3 | Saved 1 instruction vs. previous best |
| sort5 | Up to 70% faster (execution speed) |
| Sequences >250K | 1.7% faster |

- Results **merged into LLVM libc++ standard library** — now shipping in production C++ compilers
- Also improved hashing by 30%, merged into Google's **Abseil** library (billions of calls/day)

### How It Works (Assembly-Level Game)

1. Start with empty program
2. Agent adds one CPU instruction at a time
3. After each instruction, test on all possible inputs
4. Correct + shortest program wins
5. Uses Monte Carlo Tree Search (MCTS) + neural network (like AlphaGo)

### Significance

- First time RL discovered **provably faster** algorithms for a fundamental CS problem
- The improvements are at the **micro-level** (individual sort implementations for small n)
- For large arrays, existing algorithms like TimSort/IntroSort are used, calling into these improved small-sort routines

### vs. Our Thesis

- **Fundamentally different problem.** AlphaDev optimizes *how a specific sort works at the instruction level*. We optimize *which sort to choose for a given input*.
- **Complementary, not competing.** AlphaDev's improved sort3/sort5 could be components inside a hybrid selector like ours.
- Their gains are primarily for tiny sequences (3–5 elements). Our work targets macro-level selection across arbitrary-sized arrays.

---

## 3. Bai & Coester — "Sorting with Predictions" (NeurIPS 2023)

**Source:** NeurIPS 2023 (peer-reviewed), 22 pages including appendix  
**File:** `online_blogs/NeurIPS 2023 Sorting with Predictions.pdf`  
**Authors:** Xingjian Bai (Oxford), Christian Coester (Oxford)  
**Code:** https://github.com/xingjian-bai/learning-augmented-sorting

### Problem Setting

Explores sorting through the lens of **learning-augmented algorithms** — algorithms that leverage possibly erroneous predictions to improve efficiency. Two settings:

#### Setting 1: Positional Predictions
Each item aᵢ has a prediction ˆp(i) of its position in the sorted list. Error measures:
- **Displacement error:** η^Δ_i = |ˆp(i) − p(i)|
- **One-sided errors:**
  - Left-error: η^l_i = |{j: ˆp(j) ≤ ˆp(i) ∧ p(j) > p(i)}|
  - Right-error: η^r_i = |{j: ˆp(j) ≥ ˆp(i) ∧ p(j) < p(i)}|

#### Setting 2: Dirty Comparisons
A cheap-but-noisy ("dirty") comparison oracle exists alongside expensive exact ("clean") comparisons. Goal: minimize clean comparisons.
- Error: ηᵢ = number of incorrect dirty comparisons for item i

### Proposed Algorithms

#### Algorithm 1: Dirty-Clean Sort (Dirty Comparisons Setting)
- Sequentially insert elements into a BST using dirty comparisons
- Verify correctness using minimal clean comparisons
- Correct mistakes from dirty insertions
- **Complexity:** O(n log n) dirty + O(Σ log(ηᵢ + 2)) clean comparisons

#### Algorithm 2: Displacement Sort (Positional Predictions)
- Bucket sort by predicted positions, then insert into a finger tree
- **Complexity:** O(Σ log(η^Δ_i + 2)) comparisons and time

#### Algorithm 3: Double-Hoover Sort (Positional Predictions, one-sided error)
- Maintains two sorted structures L and R
- Items processed across ⌈log n⌉ rounds with increasing "insertion strength" δ
- Each item inserted into whichever structure (L or R) is faster
- Final merge of L and R in linear time
- **Complexity:** O(Σ log(min(η^l_i, η^r_i) + 2)) comparisons

### Key Properties (All Three Algorithms)

| Property | Definition | Achieved? |
|---|---|---|
| **Consistency** | O(n) comparisons when predictions are perfect | ✓ |
| **Robustness** | Never worse than O(n log n) regardless of prediction quality | ✓ |
| **Smoothness** | Degrades gracefully with prediction error | ✓ |
| **Optimality** | Matching lower bounds proven | ✓ (Theorem 1.5) |

### Experiments

- **Synthetic data (n=1,000,000):**
  - Class setting: items divided into classes, prediction = random position within class
  - Decay setting: ranking evolves over time, outdated ranking as prediction
- **Real-world:** Country population rankings (World Bank, 1960–2010, n=263)
- **Baselines:** QuickSort, MergeSort, TimSort, Odd-Even Merge Sort, Cook-Kim division
- **Results:** Proposed algorithms consistently outperform baselines across all settings

### Extensions

#### Multiple Predictors (Theorem 1.2)
- When k different dirty comparison predictors are available
- Uses HEDGE algorithm (Freund & Schapire) to select best predictor online
- Achieves O(min_p Σ log(η^p_i + 2)) clean comparisons — as good as knowing the best predictor in advance
- Bound on k is almost tight: k ≤ 2^{O(n/log n)}

#### Probabilistic Dirty Comparisons (Appendix B.1)
- Each pair (i,j) has independent error probability η_ij
- Algorithm 1 applies directly with η_i = Σ_j η_ij
- If repeated queries are independent, repeating 2k times gives O(kn log n) dirty comparisons with reduced clean comparisons via majority vote

### Key Theoretical Insights

- **Lower bound proof (Theorem 1.5):** No algorithm can do better than O(Σ log(ηᵢ + 2)) — their algorithms are optimal
- **Global error translation:** If D = total inversions, complexity is O(n log(D/n + 2)), matching known optimal adaptive sorting bounds
- **Element-wise bounds are strictly stronger** than global bounds when prediction errors are non-uniform (by Jensen's inequality)
- **Remark A.6 (Robustness):** Even with terrible predictions, the number of clean comparisons is at most that of quicksort plus O(n log log n) — the algorithm matches quicksort performance up to a factor tending to 1 as n → ∞

### Deep Learning Connection (Related Work Section)

The paper explicitly references:
- **AlphaDev (Mankowitz et al., 2023):** "recast sorting as a single-player game, training agents to perform effectively, thus uncovering faster sorting routines for short sequences"
- **Kristo et al. (2020):** Proposed learned sorting that approximates empirical CDF — related to learned index structures

### vs. Our Thesis

- **Very different approach.** They exploit predictions *within* a single sorting algorithm to reduce comparisons. We select *between complete sorting algorithms* based on input properties.
- Their "predictions" = where each element should go in sorted order. Our "predictions" = which algorithm will be fastest.
- **Theoretical CS** (comparison complexity bounds, optimality proofs) vs. **practical systems** (wall-clock time, real data, deployment).
- Their dirty comparison model is interesting for framing — our bandit's exploration is analogous to "noisy predictions" improving over time.
- Their multiple-predictor extension (Theorem 1.2) using HEDGE has philosophical parallels to our bandit.

---

## 4. Comparison Table: All Works vs. Our Thesis

| Aspect | AHS (Balasubramanian) | AlphaDev (DeepMind) | NeurIPS 2023 (Bai & Coester) | **Our Thesis** |
|---|---|---|---|---|
| **Problem** | Select sort for integer data | Optimize sort at assembly level | Reduce comparisons using predictions | Select sort for general data + online adaptation |
| **ML Method** | XGBoost (3 features) | AlphaZero RL | Theoretical (no ML training) | XGBoost (16 features) + LinUCB bandit |
| **Features** | n, k, entropy | N/A (assembly instructions) | Positional predictions per element | 16 structural O(n) features |
| **Algorithm Pool** | Counting, Radix, Quick, Insertion | Fixed sort (sort3, sort5) | Single algorithm with predictions | Introsort, Heapsort, Timsort |
| **Data Types** | Integers only | Any (at assembly level) | Any comparable | Floats + integers |
| **Online Learning** | No (static model) | No (offline RL) | No (predictor assumed given) | **Yes (LinUCB bandit)** |
| **Evaluation Scale** | Up to 10⁹ elements | LLVM benchmark suite | Up to 10⁶ + country rankings (n=263) | Real-world: 7 domains, 300+ arrays |
| **Peer Reviewed** | No (SSRN preprint) | Yes (Nature 2023) | Yes (NeurIPS 2023) | Master's thesis |
| **Theory vs Practice** | Engineering-heavy | Engineering + RL | Pure theory | Practical system |
| **Consistency guarantee** | No formal guarantee | N/A | Proven O(n) when perfect | Target via bandit convergence |
| **Robustness guarantee** | Falls back to QuickSort | N/A | Proven O(n log n) worst case | Bounded by worst single sort |

---

## 5. Key Positioning for Our Thesis

### Our Unique Contributions (vs. all related work)

1. **Richest feature space:** 16 structural features vs AHS's 3. Captures presortedness (adj_sorted_ratio), runs (runs_ratio), inversions (inversion_ratio), distribution shape (skewness, kurtosis), entropy, and more. All computed in O(n).

2. **Online adaptation via bandit:** LinUCB contextual bandit for online learning from deployment feedback. No other work has this. AHS acknowledges RL as "future work." NeurIPS 2023's HEDGE extension for multiple predictors is philosophically similar but operates at comparison level, not algorithm selection level.

3. **General comparison sorts:** We select between introsort, heapsort, timsort — general-purpose algorithms for arbitrary numeric data. AHS is limited to integer-specific sorts (Counting, Radix) that don't apply to float data.

4. **Two-tier architecture:** Offline XGBoost warm-start + online LinUCB adaptation = novel architecture that handles both cold-start and distribution shift.

5. **Designed distribution shift evaluation:** Explicit train/test distribution mismatch (train on uniform+normal, test on lognormal+exponential) to evaluate generalization — none of the other works test this.

### How to Cite Each Work

- **AHS (Balasubramanian):** Closest comparable work using ML for sort selection. Cite to contrast feature richness (3 vs 16), data type generality (integer-only vs general), and absence of online learning. Note it is not peer-reviewed.

- **AlphaDev (Mankowitz et al., Nature 2023):** Cite as complementary — optimizes sort implementations at assembly level; our work optimizes sort selection at algorithm level. The two approaches target different levels of the sorting stack.

- **NeurIPS 2023 (Bai & Coester):** Cite as theoretical state-of-the-art for learning-augmented sorting. Note they optimize comparison counts within a single algorithm using positional predictions; we optimize wall-clock time by choosing between complete algorithms. The consistency/robustness/smoothness framework they formalize is relevant to our bandit's desired properties.

### Gaps in Related Work That We Fill

1. **No work combines offline ML + online bandit** for algorithm selection
2. **AHS uses too few features** — misses structural properties critical for comparison sort selection
3. **NeurIPS 2023 is purely theoretical** — no practical deployment, no wall-clock evaluation
4. **AlphaDev targets micro-optimization** — doesn't address the macro-level algorithm selection problem
5. **No work evaluates on 7+ real-world domains** with explicit distribution shift testing

---

## 6. Additional References Found in Papers

From Balasubramanian's references:
- Al-Shargabi & Morse (2023): "Exploration of ML Techniques for Adaptive Selection of Sorting Algorithms Based on Data Characteristics"
- Shi & Li (2019): "Leyenda: An Adaptive, Hybrid Sorting Algorithm for Large Scale Data with Limited Memory"
- Jugé (2020): "Adaptive Shivers Sort: An Alternative Sorting Algorithm"

From NeurIPS 2023's references:
- Kristo et al. (2020): "The case for a learned sorting algorithm" — uses learning component to approximate empirical CDF for sorting
- Kraska et al. (2018): "The case for learned index structures" — foundational work on ML-augmented data structures
- Estivill-Castro & Wood (1992): "A survey of adaptive sorting algorithms" — classic survey on presortedness measures
- Mannila (1985): "Measures of presortedness and optimal sorting algorithms" — foundational theory on adaptive sorting
- Cook & Kim (1980): "Best sorting algorithm for nearly sorted lists" — early work on adaptive sort selection

These could be useful for the Literature Review chapter.
