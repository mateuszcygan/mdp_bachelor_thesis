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

mdp.print_mdp_details(mdp_obj1.probabilities)
approx = algorithm.my_algorithm(mdp_obj1, 25, 25, 50)
mdp.print_mdp_details(approx)


# mdp.print_mdp_sets(mdp_obj0)
# dijkstra_0 = dijkstra.dijkstra_alg(mdp_obj0.states, mdp_obj0.probabilities, "s0", "s2")
# print(dijkstra_0)

# mdp.print_mdp_sets(mdp_obj1)
# dijkstra_1 = dijkstra.dijkstra_alg(mdp_obj1.states, mdp_obj1.probabilities, "s0", "s5")
# print(dijkstra_1)

# mdp.print_mdp_sets(mdp_debug)
# dijkstra_debug = dijkstra.dijkstra_alg(
#     mdp_debug.states, mdp_debug.probabilities, "s0", "s1"
# )
# print(dijkstra_debug)

# mdp.print_mdp_sets(mdp_dijkstra)
# dijkstra_dijkstra = dijkstra.dijkstra_alg(
#     mdp_dijkstra.states, mdp_dijkstra.probabilities, "s0", "s3"
# )
# print(dijkstra_dijkstra)
