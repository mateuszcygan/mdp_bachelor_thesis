import mdp
import policies
import unittest

class TestMdpSets(unittest.TestCase):

    def setUp(self):
        return
    
    # Test if policy for 's0' changes from 'a0' to 'a1'
    # Test if policy, value function the same for value iteration with convergence and value iteration with finite horizon
    def test_dead_end_mdp(self):

        # Define an MDP with dead end
        S = ['s0', 's1', 's2']
        A = ['a0', 'a1']
        P = {
            's0' : 
                {
                    'a0' : {'s0' : 0, 's1' : 1, 's2' : 0},
                    'a1' : {'s0' : 0, 's1' : 0, 's2' : 1}
                },
            's1' :
                {
                    'a0' : {'s0' : 0, 's1' : 1, 's2' : 0}
                },
            's2' :
                {
                    'a1' : {'s0' : 0, 's1' : 0, 's2' : 1}
                }
            }
        R = {
            's0' : 
                {
                    'a0' : {'s0' : 0, 's1' : 10, 's2' : 0},
                    'a1' : {'s0' : 0, 's1' : 0, 's2' : -4}
                },
            's1' :
                {
                    'a0' : {'s0' : 0, 's1' : -1, 's2' : 0}
                },
            's2' :
                {
                    'a1' : {'s0' : 0, 's1' : 0, 's2' : 1}
                }
            }
        mdp_dead_end = mdp.MDP(S, A, P, R)

        V_dead_end_convergence, policy_dead_end_convergence = policies.value_iteration(mdp_dead_end, 0.01, 0.9)
        V_dead_end_finite, policy_dead_end_finite = policies.value_iteration(mdp_dead_end, 0.01, 0.9)
        policy = {'s0' : 'a1', 's1' : 'a0', 's2' : 'a1'}

        self.assertDictEqual(policy_dead_end_convergence, policy)
        self.assertDictEqual(policy_dead_end_finite, policy)
        self.assertDictEqual(V_dead_end_convergence, V_dead_end_finite)

    # Test policy if all rewards have equal value
    # crucial: 'max' function in Python - chooses FIRST biggest element from an array
    def test_equal_rewards_values(self):
        S = ['s0', 's1', 's2']
        A = ['a0', 'a1', 'a2', 'a3']
        R = {
            's0': 
                {'a0': {'s0': 4, 's1': 4, 's2': 4}, 'a1': {'s0': 4, 's1': 4, 's2': 4}, 'a2': {'s0': 4, 's1': 4, 's2': 4}, 'a3': {'s0': 4, 's1': 4, 's2': 4}},
            's1': 
                {'a0': {'s0': 4, 's1': 4, 's2': 4}, 'a1': {'s0': 4, 's1': 4, 's2': 4}, 'a2': {'s0': 4, 's1': 4, 's2': 4}, 'a3': {'s0': 4, 's1': 4, 's2': 4}}, 
            's2': 
                {'a0': {'s0': 4, 's1': 4, 's2': 4}, 'a1': {'s0': 4, 's1': 4, 's2': 4}, 'a2': {'s0': 4, 's1': 4, 's2': 4}, 'a3': {'s0': 4, 's1': 4, 's2': 4}}
            }
        P = {
            's0': 
                {'a0': {'s0': 0.17, 's1': 0.79, 's2': 0.04}, 'a1': {'s0': 0.71, 's1': 0.12, 's2': 0.17}, 'a2': {'s0': 0.89, 's1': 0.0, 's2': 0.11}, 'a3': {'s0': 0.38, 's1': 0.1, 's2': 0.52}},
            's1': 
                {'a0': {'s0': 0.7, 's1': 0.27, 's2': 0.03}, 'a1': {'s0': 0.58, 's1': 0.05, 's2': 0.37}, 'a2': {'s0': 0.82, 's1': 0.04, 's2': 0.14}, 'a3': {'s0': 0.79, 's1': 0.04, 's2': 0.17}},
            's2': 
                {'a0': {'s0': 0.02, 's1': 0.47, 's2': 0.51}, 'a1': {'s0': 0.78, 's1': 0.1, 's2': 0.12}, 'a2': {'s0': 0.15, 's1': 0.17, 's2': 0.68}, 'a3': {'s0': 0.79, 's1': 0.05, 's2': 0.16}}
            }

        mdp_equal_rewards = mdp.MDP(S, A, P, R)

        V, policy = policies.value_iteration_finite(mdp_equal_rewards, 1, 1)
        expected_policy = {'s0' : 'a0', 's1' : 'a0', 's2' : 'a0'}

        self.assertDictEqual(policy, expected_policy)

if __name__ == '__main__':
    unittest.main()