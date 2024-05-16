import sys

import algorithm
import dijkstra
import generator
import mdp
import policies

# Read saved mdp0
mdp_obj0 = generator.read_saved_mdp("mdp_obj0.pkl")
mdp_obj1 = generator.read_saved_mdp("mdp_obj1.pkl")
mdp_debug = generator.read_saved_mdp("mdp_debug.pkl")
mdp_dijkstra = generator.read_saved_mdp("mdp_dijkstra.pkl")

# Test
approx_prob_test, states_hits_test = algorithm.my_algo_alternating(
    mdp_obj1, 20, 0.7, 10, 0.1, 25, 25
)

states_hits = {
    "s0": {
        "a0": {"s0": 3, "s1": 12, "s2": 0},
        "a1": {"s0": 3, "s1": 2, "s2": 3},
        "a2": {"s0": 18, "s1": 0, "s2": 3},
    },
    "s1": {
        "a0": {"s0": 5, "s1": 1, "s2": 4},
        "a1": {"s0": 3, "s1": 5, "s2": 3},
        "a2": {"s0": 3, "s1": 1, "s2": 1},
    },
    "s2": {
        "a0": {"s0": 8, "s1": 0, "s2": 10},
        "a1": {"s0": 4, "s1": 1, "s2": 2},
        "a2": {"s0": 2, "s1": 1, "s2": 3},
    },
}

state_action_hits_sum = algorithm.calculate_state_action_hits(states_hits_test)
print(state_action_hits_sum)
