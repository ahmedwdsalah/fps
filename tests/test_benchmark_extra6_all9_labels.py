import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_extra6_all9_labels import counting_sort, time_algo


class BenchmarkExtra6All9LabelsTest(unittest.TestCase):
    def test_counting_sort_handles_float_values(self):
        data = [3.5, 1.2, 3.5, -4.0, 0.0, 1.2]
        self.assertEqual(counting_sort(data), sorted(data))

    def test_counting_sort_timing_does_not_fail_on_float_array(self):
        arr = np.array([3.5, 1.2, 3.5, -4.0, 0.0, 1.2], dtype=np.float64)
        seconds = time_algo(arr, "counting_sort", 1)
        self.assertTrue(math.isfinite(seconds))
        self.assertGreaterEqual(seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
