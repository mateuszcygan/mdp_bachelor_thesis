import algorithm
import generator
import mdp
import policies

# Read saved mdp0

mdp_obj0 = generator.read_saved_mdp("mdp_obj0.pkl")

# Print MDP details

# print("States:", mdp_obj0.states)
# print("Actions:", mdp_obj0.actions)
mdp.print_mdp_details(mdp_obj0.probabilities)
# mdp.print_mdp_details(mdp_obj0.rewards)

result = algorithm.my_algorithm(mdp_obj0, 25, 20)
print("Result")
mdp.print_mdp_details(result)
