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

    def test_assign_initial_probabilities(self):

        expected_probabilities = {
            "s0": {"a0": {"s0": 0.5, "s1": 0.5}, "a1": {"s0": 0.5, "s1": 0.5}},
            "s1": {"a1": {"s0": 0.5, "s1": 0.5}},
        }

        initial_probabilities = algorithm.assign_initial_probabilities(
            self.states, self.probabilities
        )

        self.assertDictEqual(expected_probabilities, initial_probabilities)

    def test_execute_action(self):

        expected_next_state = "s1"

        next_state = algorithm.execute_action(
            self.states, self.probabilities_small, "s0", "a1"
        )

        self.assertEqual(expected_next_state, next_state)


# class TestLearnProbabilities(unittest.TestCase):

#     def setUp(self):
#         self.states_num = random.randint(2, 9)
#         self.actions_num = random.randint(2, 9)
#         self.min_reward = random.randint(-8, 0)
#         self.max_reward = random.randint(1, 8)

#     # Test if sum of approximated probabilities for each state is equals (close to) 1.0
#     def test_prob_sum_finite(self):

#         for _ in range(100):  # Test for 100 different MDPs

#             mdp_object = mdp.createMDP(
#                 self.states_num, self.actions_num, self.min_reward, self.max_reward
#             )
#             iteration_number = 1000

#             result = algorithm.learn_probabilities_finite(mdp_object, iteration_number)

#             # Check if the sum of all probabilities is close to 1
#             for state, action_foll_states in result.items():
#                 for action, foll_states in action_foll_states.items():
#                     state_prob_sum = sum(list(foll_states.values()))
#                     self.assertAlmostEqual(state_prob_sum, 1.0, delta=1e-6)

#     # def test_prob_sum_convergence(self):

#     #     for _ in range(100):  # Test for 100 different MDPs

#     #         mdp_object = mdp.createMDP(
#     #             self.states_num, self.actions_num, self.min_reward, self.max_reward
#     #         )
#     #         threshold = 0.001

#     #         result = algorithm.learn_probabilities_convergence(mdp_object, threshold)

#     #         # Check if the sum of all probabilities is close to 1
#     #         for state, action_foll_states in result.items():
#     #             for action, foll_states in action_foll_states.items():
#     #                 state_prob_sum = sum(list(foll_states.values()))
#     #                 self.assertAlmostEqual(state_prob_sum, 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
