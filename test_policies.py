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

if __name__ == '__main__':
    unittest.main()