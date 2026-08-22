import unittest
import time
from src.observability.timer import Timer
from src.observability.metrics import compute_percentiles, compute_stage_breakdown
from src.observability.cache import LRUCache


class TestObservabilityAndLatency(unittest.TestCase):

    def test_timer_precision(self):
        timer = Timer()
        timer.start()
        time.sleep(0.01) # 10 ms
        elapsed = timer.stop()
        self.assertGreaterEqual(elapsed, 8.0)
        self.assertLessEqual(elapsed, 100.0)

        # Context manager usage
        with Timer() as t:
            time.sleep(0.005)
        self.assertGreater(t.elapsed_ms(), 0.0)

    def test_percentile_calculations(self):
        values = list(range(1, 101)) # 1 to 100
        stats = compute_percentiles(values)
        self.assertAlmostEqual(stats["P50"], 50.5, delta=1.0)
        self.assertAlmostEqual(stats["P70"], 70.3, delta=1.0)
        self.assertAlmostEqual(stats["P95"], 95.05, delta=1.0)
        self.assertAlmostEqual(stats["P100"], 100.0, delta=0.1)
        self.assertAlmostEqual(stats["mean"], 50.5, delta=0.5)

    def test_lru_cache_operations(self):
        cache = LRUCache(max_size=2, ttl_sec=1.0)

        # Cache miss
        self.assertIsNone(cache.get("query_a"))

        # Cache put and hit
        cache.put("query_a", "response_a")
        self.assertEqual(cache.get("query_a"), "response_a")
        self.assertGreater(cache.hit_rate, 0.0)

        # Eviction test (max_size=2)
        cache.put("query_b", "response_b")
        cache.put("query_c", "response_c") # should evict query_a
        self.assertIsNone(cache.get("query_a"))
        self.assertEqual(cache.get("query_c"), "response_c")


if __name__ == "__main__":
    unittest.main()
