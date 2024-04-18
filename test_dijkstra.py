import unittest

import dijkstra
import mdp


class TestDijkstra(unittest.TestCase):

    def setUp(self):
        self.states = ["s0", "s1", "s2", "s3"]
        self.actions = ["a0", "a1", "a2"]
        self.prob = mdp.generate_prob(self.states, self.actions)
        mdp.set_all_values_to_zero(self.prob)

        # Set probabilities for following mdp:
        # start_node "s0"
        # end_node "s3"
        # 2 options:
        # 1) "s0" -> "s3" (prob: 0.1)
        # 2) "s0" -> "s1" -> "s2" -> "s3" (prob: 0.9 * 0.9 * 0.9 = 0.729)

        self.prob["s0"]["a1"]["s3"] = 0.1
        self.prob["s0"]["a1"]["s0"] = 0.9

        self.prob["s1"]["a2"]["s2"] = 0.9
        self.prob["s1"]["a2"]["s1"] = 0.1

        self.prob["s2"]["a1"]["s3"] = 0.9
        self.prob["s2"]["a1"]["s2"] = 0.1

        # Set all other states to loop
        for state in self.states:
            for action in self.actions:
                values = mdp.get_foll_states_prob_values(self.prob, state, action)
                values_sum = sum(values)

                if values_sum == 0:
                    self.prob[state][action][state] = 1

    def test_neighbour_biggest_prob(self):
        return

    def test_probabilities(self):
        # Print the content of self.prob
        mdp.print_mdp_details(self.prob)


if __name__ == "__main__":
    unittest.main()
