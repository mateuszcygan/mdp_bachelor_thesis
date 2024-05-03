import random
import unittest

import algorithm
import mdp


class TestEssentialFunctions(unittest.TestCase):

    def setUp(self):

        # Define MDP sets for initial probabilities test
        self.states = ["s0", "s1"]
        self.actions = ["a0", "a1"]
        self.probabilities = {
            "s0": {"a0": {"s0": 0.75, "s1": 0.25}, "a1": {"s0": 0.43, "s1": 0.57}},
            "s1": {"a1": {"s0": 0.4, "s1": 0.6}},
        }
        self.rewards = {
            "s0": {"a0": {"s0": 3, "s1": 4}, "a1": {"s0": 0, "s1": -2}},
            "s1": {"a1": {"s0": 2, "s1": -3}},
        }

        # Define MDP's probabilities set for execution of an action
        self.probabilities_small = {
            "s0": {"a0": {"s0": 1.0, "s1": 0}, "a1": {"s0": 0, "s1": 1.0}},
            "s1": {"a1": {"s0": 0.4, "s1": 0.6}},
        }
        self.states_hits = algorithm.create_states_hits_dictionary(self.probabilities)

        # Define states hits dictionary for testing purposes
        self.states_hits = {
            "s0": {
                "a0": {"s0": 5, "s1": 14, "s2": 3},
                "a1": {"s0": 32, "s1": 4, "s2": 23},
                "a2": {"s0": 2, "s1": 0, "s2": 1},
            },
            "s1": {
                "a0": {"s0": 11, "s1": 9, "s2": 6},
                "a1": {"s0": 12, "s1": 7, "s2": 8},
                "a2": {"s0": 0, "s1": 3, "s2": 0},
            },
            "s2": {
                "a0": {"s0": 1, "s1": 1, "s2": 1},
                "a1": {"s0": 20, "s1": 17, "s2": 12},
                "a2": {"s0": 0, "s1": 1, "s2": 2},
            },
        }

        self.state_action_hits = {
            "s0": {"a0": 22, "a1": 59, "a2": 3},
            "s1": {"a0": 26, "a1": 27, "a2": 3},
            "s2": {"a0": 3, "a1": 49, "a2": 3},
        }

    # Test if initial probabilities are initially correct assigned (equal probabilities dependent on states' number)
    def test_assign_initial_approx_probabilities(self):

        expected_probabilities = {
            "s0": {"a0": {"s0": 0.5, "s1": 0.5}, "a1": {"s0": 0.5, "s1": 0.5}},
            "s1": {"a1": {"s0": 0.5, "s1": 0.5}},
        }

        initial_probabilities = algorithm.assign_initial_approx_probabilities(
            self.states, self.probabilities
        )

        self.assertDictEqual(expected_probabilities, initial_probabilities)

    # Test if next state will be the one with probability assigned to 1.0
    def test_execute_action_small(self):

        expected_next_state = "s1"

        next_state, states_hits = algorithm.execute_action(
            self.states, self.probabilities_small, "s0", "a1", self.states_hits
        )

        self.assertEqual(expected_next_state, next_state)

    # Test if the approximated probability updates correctly (MDP with 4 states chosen due to calculation of new probability - 0.25 to 0.2 and one to 0.4)
    def test_update_approx_prob(self):

        # Create a MDP with 4 states and 2 actions
        test_mdp = mdp.createMDP(4, 2, 0, 3)

        # Create elements for "update_approx_prob" function
        approximated_prob = algorithm.assign_initial_approx_probabilities(
            test_mdp.states, test_mdp.probabilities
        )
        states_hits = {
            s: {a: {s: 0 for s in test_mdp.states} for a in test_mdp.actions}
            for s in test_mdp.states
        }
        current_state = "s1"
        executed_action = "a1"

        ## Assume that after execution of executed_action, the transition took place to s1
        states_hits[current_state][executed_action]["s1"] = 1

        approximated_prob = algorithm.update_approx_prob(
            approximated_prob,
            states_hits,
            current_state,
            executed_action,
            test_mdp.states,
        )

        expected_approx_prob = {
            "s0": {
                "a0": {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25},
                "a1": {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25},
            },
            "s1": {
                "a0": {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25},
                "a1": {"s0": 0.2, "s1": 0.4, "s2": 0.2, "s3": 0.2},
            },
            "s2": {
                "a0": {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25},
                "a1": {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25},
            },
            "s3": {
                "a0": {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25},
                "a1": {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25},
            },
        }

        self.assertDictEqual(approximated_prob, expected_approx_prob)

    # Checks if for each state a certain number of states' hits took place
    def test_check_desired_states_hits_num(self):

        result1 = algorithm.check_desired_states_hits_num(self.states_hits, 6)
        self.assertTrue(result1)

        result2 = algorithm.check_desired_states_hits_num(self.states_hits, 100)
        self.assertFalse(result2)

    def test_calculate_state_action_hits(self):

        result = algorithm.calculate_state_action_hits(self.states_hits)
        self.assertDictEqual(result, self.state_action_hits)


class TestLearnProbabilities(unittest.TestCase):

    def setUp(self):
        self.states_num = random.randint(2, 9)
        self.actions_num = random.randint(2, 9)
        self.min_reward = random.randint(-8, 0)
        self.max_reward = random.randint(1, 8)

    # Test if sum of approximated probabilities for each state is equals (close to) 1.0
    def test_my_algorithm(self):

        for _ in range(100):  # Test for 100 different MDPs

            mdp_object = mdp.createMDP(
                self.states_num, self.actions_num, self.min_reward, self.max_reward
            )

            S = mdp_object.states
            P = mdp_object.probabilities
            approx_prob = algorithm.assign_initial_approx_probabilities(S, P)
            states_hits = algorithm.create_states_hits_dictionary(P)
            current_state = "s0"
            iteration_number = 100

            result, result_states_hits = algorithm.my_algorithm(mdp_object, 25, 25, 25)

            # Check if the sum of all probabilities is close to 1
            for state, action_foll_states in result.items():
                for action, foll_states in action_foll_states.items():
                    state_prob_sum = sum(list(foll_states.values()))
                    self.assertAlmostEqual(state_prob_sum, 1.0, delta=1e-6)


class TestConvergence(unittest.TestCase):

    def setUp(self):
        self.probabilities_old = {
            "s0": {"a0": {"s0": 0.75, "s1": 0.25}, "a1": {"s0": 0.43, "s1": 0.57}},
            "s1": {"a1": {"s0": 0.4, "s1": 0.6}},
        }

        self.probabilities_new = {
            "s0": {"a0": {"s0": 0.74, "s1": 0.24}, "a1": {"s0": 0.43, "s1": 0.57}},
            "s1": {"a1": {"s0": 0.6, "s1": 0.4}},
        }

        self.threshold1 = 0.1
        self.threshold2 = 0.2

    def test_check_specific_prob_convergence(self):

        prob_to_check = [("s0", "a0"), ("s1", "a1")]

        result1 = algorithm.check_specific_prob_convergence(
            self.probabilities_old,
            self.probabilities_new,
            prob_to_check,
            self.threshold1,
        )
        result2 = algorithm.check_specific_prob_convergence(
            self.probabilities_old,
            self.probabilities_new,
            prob_to_check,
            self.threshold2,
        )

        self.assertFalse(result1)
        self.assertTrue(result2)

    def test_check_prob_convergence(self):

        states = ["s0", "s1"]
        result1 = algorithm.check_prob_convergence(
            states, self.threshold1, self.probabilities_old, self.probabilities_new
        )
        result2 = algorithm.check_prob_convergence(
            states, self.threshold2, self.probabilities_old, self.probabilities_new
        )

        self.assertFalse(result1)
        self.assertTrue(result2)


if __name__ == "__main__":
    unittest.main()
