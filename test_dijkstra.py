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
        self.prob["s0"]["a0"]["s1"] = 0.9
        self.prob["s0"]["a0"]["s0"] = 0.1

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
        unvisited_states = self.states

        s0_neighbour_s1 = dijkstra.NeighbourProbabilityEntry("a0", 0.9)
        s0_neighbour_s3 = dijkstra.NeighbourProbabilityEntry("a1", 0.1)

        s0_neighbours = {"s1": s0_neighbour_s1, "s3": s0_neighbour_s3}

        s0_result = dijkstra.neighbour_biggest_prob(unvisited_states, self.prob, "s0")

        for key, values in s0_neighbours.items():
            self.assertEqual(values.action, s0_result[key].action)
            self.assertEqual(values.probability, s0_result[key].probability)

        unvisited_states.remove("s0")

        # Go over to s1 and extract neighbours
        s1_neighbour_s2 = dijkstra.NeighbourProbabilityEntry("a2", 0.9)
        s1_neighbours = {"s2": s1_neighbour_s2}

        s1_result = dijkstra.neighbour_biggest_prob(unvisited_states, self.prob, "s1")

        for key, values in s1_neighbours.items():
            self.assertEqual(values.action, s1_result[key].action)
            self.assertEqual(values.probability, s1_result[key].probability)

        unvisited_states.remove("s1")

        # Go over to s2 and extract neighbours
        s2_neighbour_s3 = dijkstra.NeighbourProbabilityEntry("a1", 0.9)
        s2_neighbours = {"s3": s2_neighbour_s3}

        s2_result = dijkstra.neighbour_biggest_prob(unvisited_states, self.prob, "s2")

        for key, values in s2_neighbours.items():
            self.assertEqual(values.action, s2_result[key].action)
            self.assertEqual(values.probability, s2_result[key].probability)

        # Check if not reachable states are deleted
        s3_neighbours = {}
        s3_result = dijkstra.neighbour_biggest_prob(self.states, self.prob, "s3")
        self.assertDictEqual(s3_neighbours, s3_result)

    # Function for printing purposes
    def test_print(self):
        # Print the content of self.prob
        mdp.print_mdp_details(self.prob)


if __name__ == "__main__":
    unittest.main()
