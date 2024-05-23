import sys

import algorithm
import dijkstra
import generator
import mdp
import mdp_strategies
import policies

# Read saved mdp0
mdp_obj0 = generator.read_saved_mdp("mdp_obj0.pkl")
mdp_obj1 = generator.read_saved_mdp("mdp_obj1.pkl")
mdp_debug = generator.read_saved_mdp("mdp_debug.pkl")
mdp_dijkstra = generator.read_saved_mdp("mdp_dijkstra.pkl")

# ITERATIONS_NUM_STRATEGY

# print("mdp_obj0 probabilities:")
# mdp.print_mdp_details(mdp_obj0.probabilities)

# approximated_mdp_iterations, rewards_sum_iterations = (
#     mdp_strategies.iterations_num_strategy(
#         mdp_obj0, 200, 0.5, 25, 25, 0.3, 10, None, 0.01, 0.9, 0.2
#     )
# )

# print("mdp_obj0 approximated probabilities:")
# mdp.print_mdp_details(approximated_mdp_iterations.probabilities)

# print("mdp_obj0 learned rewards:")
# mdp.print_mdp_details(approximated_mdp_iterations.rewards)

# print("rewards_sum:", rewards_sum_iterations)

# ITERATIONS_NUM_STRATEGY


# MDP_KNOWLEDGE_STRATEGY
approximated_mdp_knowledge, rewards_sum_knowledge = (
    mdp_strategies.mdp_knowledge_strategy(
        mdp_obj1, 400, 0.9, 25, 25, 0.3, 10, None, 0.01, 0.9, 0.2
    )
)

print("mdp_obj1 approximated probabilities:")
mdp.print_mdp_details(approximated_mdp_knowledge.probabilities)

print("mdp_obj1 learned rewards:")
mdp.print_mdp_details(approximated_mdp_knowledge.rewards)

print("rewards_sum:", rewards_sum_knowledge)


# MDP_KNOWLEDGE_STRATEGY
