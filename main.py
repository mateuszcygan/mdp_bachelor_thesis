import mdp
import generator
import policies
import algorithm

# Create MDP

# mdp_obj = mdp.createMDP()
# print("States:", mdp_obj.states)
# print("Actions:", mdp_obj.actions)
# mdp.print_mdp_details(mdp_obj.probabilities)
# mdp.print_mdp_details(mdp_obj.rewards)



# Read saved mdp0

mdp_obj0 = generator.read_saved_mdp("mdp_obj0.pkl")

# Print MDP details

# print("States:", mdp_obj0.states)
# print("Actions:", mdp_obj0.actions)
# mdp.print_mdp_details(mdp_obj0.probabilities)
# mdp.print_mdp_details(mdp_obj0.rewards)



# Sparse states

# mdp.unreachable_states(mdp_obj0)
# mdp.sparse_mdp_states(mdp_obj0, 0.2)
# mdp.print_mdp_details(mdp_obj0.probabilities)
# mdp.unreachable_states(mdp_obj0)



# Sparse rewards

# mdp.zero_rewards(mdp_obj0)
# mdp.sparse_mdp_rewards(mdp_obj0, 0.25)
# mdp.print_mdp_details(mdp_obj0.rewards)
# mdp.zero_rewards(mdp_obj0)



# Reduce actions number

# mdp.reduce_actions_number(mdp_obj0, 1, 3)
# mdp.print_mdp_details(mdp_obj0.probabilities)
# mdp.print_mdp_details(mdp_obj0.rewards)



# Read saved reduced mdp0
mdp_obj1 = generator.read_saved_mdp("mdp_obj1.pkl")



# Print MDP details

# print("States:", mdp_obj1.states)
# print("Actions:", mdp_obj1.actions)
# print()
# mdp.print_mdp_details(mdp_obj1.probabilities)
# mdp.print_mdp_details(mdp_obj1.rewards)
# mdp.unreachable_states(mdp_obj1)
# mdp.zero_rewards(mdp_obj1)



# Random policy

# print("Random policy\n")
# random_policy = policies.random_policy(mdp_obj1, 10)
# print("Random policy:", random_policy)



# Value iteration

# mdp_obj0
# V0, policy0 = policies.value_iteration(mdp_obj0, 0.0001)
# print("Policy with convergence:", policy0)
# print("Value function:", V0)

# print("\n")

# V_finite0, policy_finite0 = policies.value_iteration_finite(mdp_obj0, 91)
# print("Policy with finite horizon:", policy_finite0)
# print("Value function:", V_finite0)


# mdp_obj1
# V1, policy1 = policies.value_iteration(mdp_obj1, 0.001)
# print("Policy with convergence:", policy1)
# print("Value function:", V1)

# V_finite1, policy_finite1 = policies.value_iteration_finite(mdp_obj1, 100)
# print("Policy with finite horizon:", policy_finite1)
# print("Value function:", V_finite1)

# Value iteration - value for s1 = 0
# P = mdp_obj1.probabilities
# R = mdp_obj1.rewards
# S = mdp_obj1.states

# current_state = "s1"
# executable_actions = mdp.get_possible_actions(P, current_state)
# for action in executable_actions:
#     print("CURRENT ACTION:", action)
#     for state in S:
#         print("prob state", state, ":", P[current_state][action][state])
#         print("reward state", state, ":", R[current_state][action][state])
#         print("\n")



# Algorithm

# mdp_obj0
# mdp.print_mdp_details(mdp_obj0.probabilities)

# approx_prob0 = algorithm.learn_probabilities_convergence(mdp_obj0, 0.01)
# mdp.print_mdp_details(approx_prob0)

# approx_prob_finite0 = algorithm.learn_probabilities_finite(mdp_obj0, 192)
# mdp.print_mdp_details(approx_prob_finite0)

# mdp_obj1
# mdp.print_mdp_details(mdp_obj1.probabilities)

# approx_prob1 = algorithm.learn_probabilities_convergence(mdp_obj1, 0.01)
# mdp.print_mdp_details(approx_prob1)

# approx_prob_finite1 = algorithm.learn_probabilities_finite(mdp_obj1, 192)
# mdp.print_mdp_details(approx_prob_finite1)