import mdp
import random

# RANDOM POLICY

# Based on selecting randomly action that should be executed
# 1. Select randomly action
# 2. Based on probabilities, going over to another state will be executed

def random_policy(mdp_object, tran_num):

    # Sets of which MDP is composed (replace with new function that is already implemented)
    S, A, R, P = mdp_object.get_properties()

    current_state = 's0' # Start from s0
    rewards_sum = 0

    policy = {s : None for s in S}

    for _ in range(0, tran_num): # Use _ as a placeholder for the loop variable

        print("curr_state:", current_state)
        executable_actions = mdp.get_possible_actions(P, current_state)
        executed_action = random.choice(executable_actions) # Randomly choose action from A
        
        print("random_action:", executed_action)
        policy[current_state] = executed_action # Add chosen action to the policy

        states_weights = mdp.get_foll_states_prob_values(P, current_state, executed_action) # Extract probabilities for following states
        print("tran_prob", current_state, ":", states_weights)

        states_rewards = mdp.get_foll_states_rewards_values(R, current_state, executed_action)
        print("rewards_foll_states:", states_rewards) # Print rewards assigned to each of the following states

        # Go over to another state based on transition probabilities
        next_state = random.choices(S, weights = states_weights)[0]
        print("foll_state(prob_based):", next_state, '\n')

        reward = R[current_state][executed_action][next_state]
        rewards_sum += reward

        print("reward:", reward)
        print("collected_rewards:", rewards_sum)
        print("\n\n")

        current_state = next_state

    return policy



# VALUE ITERATION
# Printing formatted version of result from value iteration
def print_value_function(V):
    for state, value in V.items():
        print(state, ":", value)

def value_iteration(mdp_object):

    # Extract MDP properties
    S, A, R, P = mdp_object.get_properties()

    dicount_factor = 0.9

    # Initialize the value for each state to 0
    V = {s : 0 for s in S}
    V_new = {s : 0 for s in S}
    policy = {s : None for s in S} # Dictionary that represents calculated policy

    i = 0 # Number of iterations

    while True:
        i += 1
        print("Iteration number", i, "\n")

        for current_state in S:

            actions_values = {action : 0 for action in A} # Dictionary with value for each action is a specific state

            for a in A:

                # Get probabilities and rewards for current state
                probabilities = list(P[current_state][a].values())
                rewards = list(R[current_state][a].values())

                # Calculate sum(Pa(s,s')(Ra(s,s') + dicount_factor*V_i(s')))
                value = sum([prob * (reward + dicount_factor * V[current_state]) for prob, reward in zip(probabilities, rewards)])
                actions_values[a] = value
            
            print("Calculated values for state", current_state, ":", actions_values)

            # Find the biggest value and its action
            max_value = max(list(actions_values.values()))
            biggest_value_action = list(actions_values.keys())[list(actions_values.values()).index(max_value)]

            policy[current_state] = biggest_value_action # Set the action with the biggest value as the policy action from current state

            V_new[current_state] = max_value # Set maximum of calculated values as a new value for current_state

            # Termination condition
            print("Old value function:")
            print_value_function(V)
            print('\n')
            print("New value function:")
            print_value_function(V_new)
            print("\n")
            if (all( abs(V[s] - V_new[s]) < 0.001 for s in S)):
                print("Number of iterations: ", i)
                return V_new, policy
            V = V_new.copy()

def value_iteration_finite(mdp_object, finite_horizon):

    # Extract MDP properties
    S, A, R, P = mdp_object.get_properties()

    dicount_factor = 0.9 # For comparisson
    # discount_factor = 1

    # Initialize the value for each state to 0
    V = {s : 0 for s in S}
    V_new = {s : 0 for s in S}
    policy = {s : None for s in S} # Dictionary that represents calculated policy

    iterations = range(finite_horizon) # Number of iterations
    i = 0


    for x in iterations:
        print("Value of x:", x)
        i += 1

        for current_state in S:

            actions_values = {action : 0 for action in A} # Dictionary with value for each action is a specific state

            for a in A:

                # Get probabilities and rewards for current state
                probabilities = list(P[current_state][a].values())
                rewards = list(R[current_state][a].values())

                # Calculate sum(Pa(s,s')(Ra(s,s') + dicount_factor*V_i(s')))
                value = sum([prob * (reward + dicount_factor * V[current_state]) for prob, reward in zip(probabilities, rewards)])
                actions_values[a] = value
            
            print("Calculated values for state", current_state, ":", actions_values)

            # Find the biggest value and its action
            max_value = max(list(actions_values.values()))
            biggest_value_action = list(actions_values.keys())[list(actions_values.values()).index(max_value)]

            policy[current_state] = biggest_value_action # Set the action with the biggest value as the policy action from current state

            V_new[current_state] = max_value # Set maximum of calculated values as a new value for current_state

            print("Old value function:")
            print_value_function(V)
            print('\n')
            print("New value function:")
            print_value_function(V_new)
            print("\n")

            V = V_new.copy()
        
    return V_new, policy