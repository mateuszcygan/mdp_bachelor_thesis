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

# mdp.print_mdp_sets(mdp_debug)

algorithm.my_algo_alternating(mdp_debug, 4, 0.1, 5, 25, 25)
mdp.print_mdp_details(mdp_debug.probabilities)


# Test
approx_prob_test, states_hits_test = algorithm.my_algo_alternating(
    mdp_obj0,
    4,
    0.001,
    8,
    10,
    25,
)
