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
        self.prob["s0"]["a1"]["s1"] = 0.9

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

        # Start - neighbours of s0, all states unvisited
        s0_neighbours = {"s1": {"a1": 0.9}, "s3": {"a1": 0.1}}

        unvisited_states = self.states
        s0_result = dijkstra.neighbour_biggest_prob(unvisited_states, self.prob, "s0")

        self.assertDictEqual(s0_neighbours, s0_result)

        unvisited_states.remove("s0")

        # Go over to s1 and extract neighbours
        s1_neighbours = {"s2": {"a2": 0.9}}

        s1_result = dijkstra.neighbour_biggest_prob(unvisited_states, self.prob, "s1")

        self.assertDictEqual(s1_neighbours, s1_result)

        unvisited_states.remove("s1")

        # Go over to s2 and extract neighbours
        s2_neighbours = {"s3": {"a1": 0.9}}

        s2_result = dijkstra.neighbour_biggest_prob(unvisited_states, self.prob, "s2")
        self.assertDictEqual(s2_neighbours, s2_result)

        # Check if not reachable states are deleted
        s3_neighbours = {}
        s3_result = dijkstra.neighbour_biggest_prob(self.states, self.prob, "s3")
        self.assertDictEqual(s3_neighbours, s3_result)

    def test_probabilities(self):
        # Print the content of self.prob
        mdp.print_mdp_details(self.prob)


if __name__ == "__main__":
    unittest.main()
