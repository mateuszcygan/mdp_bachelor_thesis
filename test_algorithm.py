import unittest
import algorithm
import mdp

class TestLearnProbabilities(unittest.TestCase):

    # Test if sum of approximated probabilities for each state is equals (close to) 1.0
    
    def test_prob_sum_finite(self):

        for _ in range(100): # Test for 100 different MDPs

            mdp_object = mdp.createMDP()
            iteration_number = 1000

            result = algorithm.learn_probabilities_finite(mdp_object, iteration_number)

            # Check if the sum of all probabilities is close to 1
            for state, action_foll_states in result.items():
                for action, foll_states in action_foll_states.items():
                    state_prob_sum = sum(list(foll_states.values()))
                    self.assertAlmostEqual(state_prob_sum, 1.0, delta=1e-6)
    
    def test_prob_sum_convergence(self):

        for _ in range(100): # Test for 100 different MDPs

            mdp_object = mdp.createMDP()
            threshold = 0.001

            result = algorithm.learn_probabilities_convergence(mdp_object, threshold)

            # Check if the sum of all probabilities is close to 1
            for state, action_foll_states in result.items():
                for action, foll_states in action_foll_states.items():
                    state_prob_sum = sum(list(foll_states.values()))
                    self.assertAlmostEqual(state_prob_sum, 1.0, delta=1e-6)

if __name__ == '__main__':
    unittest.main()
