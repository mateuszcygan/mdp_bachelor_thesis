import copy
import sys
import unittest

import dijkstra
import mdp


def asserEqual_shortest_path(self, table1, table2):
    for key, values in table1.items():
        self.assertAlmostEqual(values.probability, table2[key].probability, delta=1e-6)
        self.assertEqual(values.previous_state, table2[key].previous_state)
        self.assertEqual(
            values.executed_action_in_prev_state,
            table2[key].executed_action_in_prev_state,
        )


# For states where all probabilities are equal 0, the probabilty equal 1.0 is set for the set from which action is executed (loop state)
def set_loop_states(states, actions, probabilities):
    for state in states:
        for action in actions:
            values = mdp.get_foll_states_prob_values(probabilities, state, action)
            values_sum = sum(values)

            if values_sum == 0:
                probabilities[state][action][state] = 1


class TestDijkstra(unittest.TestCase):

    def setUp(self):

        # Specification of first mdp to test
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
        set_loop_states(self.states, self.actions, self.prob)

        # Specification of second mdp to test
        self.states1 = ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]
        self.actions1 = ["a0", "a1", "a2"]
        self.prob1 = mdp.generate_prob(self.states1, self.actions1)
        mdp.set_all_values_to_zero(self.prob1)

        self.prob1["s0"]["a0"]["s1"] = 0.9
        self.prob1["s0"]["a0"]["s0"] = 0.1
        self.prob1["s0"]["a1"]["s2"] = 0.3
        self.prob1["s0"]["a1"]["s0"] = 0.7

        self.prob1["s1"]["a1"]["s3"] = 0.85
        self.prob1["s1"]["a1"]["s1"] = 0.15
        self.prob1["s1"]["a2"]["s4"] = 0.9
        self.prob1["s1"]["a2"]["s1"] = 0.1

        self.prob1["s2"]["a0"]["s5"] = 0.5
        self.prob1["s2"]["a0"]["s2"] = 0.5
        self.prob1["s2"]["a1"]["s6"] = 0.9
        self.prob1["s2"]["a1"]["s2"] = 0.1

        self.prob1["s3"]["a1"]["s5"] = 0.85
        self.prob1["s3"]["a1"]["s3"] = 0.15

        self.prob1["s4"]["a2"]["s5"] = 0.9
        self.prob1["s4"]["a2"]["s4"] = 0.1

        set_loop_states(self.states1, self.actions1, self.prob1)
        self.rewards1 = mdp.generate_rewards(self.states1, self.actions1, -5, 5)

        self.mdp_obj1 = mdp.MDP(self.states1, self.actions1, self.prob1, self.rewards1)

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

    def test_update_shortest_path_table(self):

        start_state = "s0"
        current_visit_state = start_state
        end_state = "s3"
        shortest_path_value = -1
        shortest_path = dijkstra.create_shortest_path_table(self.states, start_state)
        unvisited_states = self.states

        neighbours = dijkstra.neighbour_biggest_prob(
            unvisited_states, self.prob, current_visit_state
        )

        result_s0, shortest_path_value = dijkstra.update_shortest_path(
            start_state,
            end_state,
            current_visit_state,
            shortest_path_value,
            shortest_path,
            neighbours,
        )

        shortest_path_s0 = copy.deepcopy(shortest_path)

        s0_neighbour_s1 = dijkstra.ShortestPathEntry(0.9, "s0", "a0")
        s0_neighbour_s3 = dijkstra.ShortestPathEntry(0.1, "s0", "a1")

        shortest_path_s0["s1"] = s0_neighbour_s1
        shortest_path_s0["s3"] = s0_neighbour_s3

        asserEqual_shortest_path(self, shortest_path_s0, result_s0)

        # Remove the state "s0" from unvisited states
        unvisited_states.remove("s0")

        # Check if "s1" is correctly chosen as next visited state
        new_current_state = dijkstra.choose_next_state_to_visit(
            shortest_path_s0, unvisited_states
        )
        current_visit_state = "s1"
        self.assertEqual(current_visit_state, new_current_state)

        # Check if the shortest path is correctly updated after transitioning to "s1"
        neighbours = dijkstra.neighbour_biggest_prob(
            unvisited_states, self.prob, current_visit_state
        )

        result_s1, shortest_path_value = dijkstra.update_shortest_path(
            start_state,
            end_state,
            current_visit_state,
            shortest_path_value,
            shortest_path_s0,
            neighbours,
        )

        s1_shortest_path_s2 = dijkstra.ShortestPathEntry(0.81, "s1", "a2")
        shortest_path_s1 = shortest_path_s0
        shortest_path_s1["s2"] = s1_shortest_path_s2

        asserEqual_shortest_path(self, shortest_path_s1, result_s1)

        # Remove the state "s1" from unvisited states
        unvisited_states.remove("s1")

        # Check if "s2" is correctly chosen as next visited state
        new_current_state = dijkstra.choose_next_state_to_visit(
            shortest_path_s1, unvisited_states
        )
        current_visit_state = "s2"
        self.assertEqual(current_visit_state, new_current_state)

        neighbours = dijkstra.neighbour_biggest_prob(
            unvisited_states, self.prob, current_visit_state
        )

        result_s2, shortest_path_value = dijkstra.update_shortest_path(
            start_state,
            end_state,
            current_visit_state,
            shortest_path_value,
            shortest_path_s1,
            neighbours,
        )

        s2_shortest_path_s3 = dijkstra.ShortestPathEntry(0.729, "s2", "a3")
        shortest_path_s2 = shortest_path_s1
        shortest_path_s2["s3"] = s2_shortest_path_s3

        asserEqual_shortest_path(self, shortest_path_s2, result_s2)

    def test_dijkstra_alg(self):
        max_value = -sys.maxsize - 1
        result = dijkstra.dijkstra_alg(self.mdp_obj1, self.prob1, "s0", "s5")

        s0_shortest_path_s0 = dijkstra.ShortestPathEntry(1, None, None)
        s0_shortest_path_s1 = dijkstra.ShortestPathEntry(0.9, "s0", "a0")
        s0_shortest_path_s2 = dijkstra.ShortestPathEntry(0.3, "s0", "a1")
        s0_shortest_path_s3 = dijkstra.ShortestPathEntry(0.765, "s1", "a1")
        s0_shortest_path_s4 = dijkstra.ShortestPathEntry(0.81, "s1", "a2")
        s0_shortest_path_s5 = dijkstra.ShortestPathEntry(0.729, "s4", "a2")
        s0_shortest_path_s6 = dijkstra.ShortestPathEntry(max_value, None, None)

        shortest_path_s0_s5 = {}
        shortest_path_s0_s5["s0"] = s0_shortest_path_s0
        shortest_path_s0_s5["s1"] = s0_shortest_path_s1
        shortest_path_s0_s5["s2"] = s0_shortest_path_s2
        shortest_path_s0_s5["s3"] = s0_shortest_path_s3
        shortest_path_s0_s5["s4"] = s0_shortest_path_s4
        shortest_path_s0_s5["s5"] = s0_shortest_path_s5
        shortest_path_s0_s5["s6"] = s0_shortest_path_s6

        asserEqual_shortest_path(self, result, shortest_path_s0_s5)

    # Function for printing purposes
    def test_print(self):
        return


if __name__ == "__main__":
    unittest.main()
