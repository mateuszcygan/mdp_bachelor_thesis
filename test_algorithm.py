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


class TestLearnProbabilities(unittest.TestCase):

    def setUp(self):
        self.states_num = random.randint(2, 9)
        self.actions_num = random.randint(2, 9)
        self.min_reward = random.randint(-8, 0)
        self.max_reward = random.randint(1, 8)

    # Test if sum of approximated probabilities for each state is equals (close to) 1.0
    def test_prob_sum_finite(self):

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

            result, current_state = algorithm.systematic_learning(
                S, P, approx_prob, states_hits, current_state, iteration_number
            )

            # Check if the sum of all probabilities is close to 1
            for state, action_foll_states in result.items():
                for action, foll_states in action_foll_states.items():
                    state_prob_sum = sum(list(foll_states.values()))
                    self.assertAlmostEqual(state_prob_sum, 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
