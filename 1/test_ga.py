import math
import unittest

from ga import michalewicz, run_ga


class MichalewiczTests(unittest.TestCase):
    def test_known_one_dimensional_value(self):
        self.assertAlmostEqual(michalewicz([2.20]), -0.8013, places=3)

    def test_seed_reproduces_result(self):
        first = run_ga(
            seed=15,
            dimension=2,
            population_size=30,
            generations=8,
            mutation_probability=0.15,
        )
        second = run_ga(
            seed=15,
            dimension=2,
            population_size=30,
            generations=8,
            mutation_probability=0.15,
        )
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_evaluations_and_bounds(self):
        point, history, _, evaluations = run_ga(
            seed=7,
            dimension=9,
            population_size=30,
            generations=5,
            mutation_probability=0.1,
        )
        self.assertEqual(evaluations, 30 * 6)
        self.assertEqual(len(history), 6)
        self.assertTrue(all(0 <= value <= math.pi for value in point))
        self.assertTrue(all(row["best"] <= row["worst"] for row in history))


if __name__ == "__main__":
    unittest.main()
