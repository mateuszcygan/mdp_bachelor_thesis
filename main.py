import mdp
import generator
import policies

# Create MDP

# mdp_obj = mdp.createMDP()
# print(mdp_obj.states)
# print(mdp_obj.actions)
# mdp.print_mdp_details(mdp_obj.probabilities)
# mdp.print_mdp_details(mdp_obj.rewards)



# Read saved mdp0

# mdp_obj0 = generator.read_saved_mdp("mdp0.pkl")

# Print MDP details

# print(mdp_obj0.states)
# print(mdp_obj0.actions)
# mdp.print_mdp_details(mdp_obj0.probabilities)
# mdp.print_mdp_details(mdp_obj0.rewards)



# Sparse states

# mdp.unreachable_states(mdp_obj0)
# mdp.sparse_mdp_states(mdp_obj0, 0.25)
# mdp.print_mdp_details(mdp_obj0.probabilities)
# mdp.unreachable_states(mdp_obj0)



# Sparse rewards

# mdp.zero_rewards(mdp_obj0)
# mdp.sparse_mdp_rewards(mdp_obj0, 0.25)
# mdp.print_mdp_details(mdp_obj0.rewards)
# mdp.zero_rewards(mdp_obj0)



# Reduce actions number

# mdp.reduce_actions_number(mdp_obj0, 3, 3)
# mdp.print_mdp_details(mdp_obj0.probabilities)
# mdp.print_mdp_details(mdp_obj0.rewards)



# Read saved mdp1
# mdp_obj1 = generator.read_saved_mdp("mdp1.pkl")

# Print MDP details
# print(mdp_obj1.states)
# print(mdp_obj1.actions)
# mdp.print_mdp_details(mdp_obj1.probabilities)
# mdp.print_mdp_details(mdp_obj1.rewards)












# print("Random policy\n")
# random_policy = policies.random_policy(mdp_obj, 10)
# print("Random policy:", random_policy)


# V, policy = policies.value_iteration(mdp_obj)
# V_finite, policy_finite = policies.value_iteration_finite(mdp_obj, 75)

# print("Value policy with convergence:\n")
# print("Value function:")
# policies.print_value_function(V)
# print("Policy:\n")
# print(policy)
# print()
# print()
# print()
# print("Value policy with finite horizon:\n")
# print("Value function:")
# policies.print_value_function(V_finite)
# print("Policy:\n")
# print(policy_finite)