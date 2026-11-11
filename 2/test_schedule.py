import random
import unittest

from schedule import build_data, objective, repair, run_ga, violations


class DutyRosterTests(unittest.TestCase):
    def setUp(self):
        self.data = build_data()

    def test_variant_dataset_size_and_conflicts(self):
        self.assertEqual(len(self.data["shifts"]), 21)
        self.assertGreaterEqual(len(self.data["hard_conflicts"]), 10)

    def test_repair_produces_feasible_roster(self):
        chromosome = [index % 10 for index in range(len(self.data["shifts"]))]
        repaired = repair(chromosome, self.data, random.Random(42))
        self.assertEqual(violations(repaired, self.data["hard_conflicts"], 10), 0)
        self.assertEqual(objective(repaired, self.data, "repair")[1], 0)

    def test_repair_is_reproducible(self):
        chromosome = [index % 10 for index in range(len(self.data["shifts"]))]
        self.assertEqual(
            repair(chromosome, self.data, random.Random(9)),
            repair(chromosome, self.data, random.Random(9)),
        )

    def test_ga_returns_feasible_schedule_with_repair(self):
        schedule, history, evaluations = run_ga(
            data=self.data,
            seed=15,
            population_size=30,
            generations=4,
            strategy="repair",
            mutation="reassign",
        )
        self.assertEqual(len(history), 5)
        self.assertEqual(evaluations, 30 * 5)
        self.assertEqual(violations(schedule, self.data["hard_conflicts"], 10), 0)


if __name__ == "__main__":
    unittest.main()
