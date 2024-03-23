import random
import copy
import mdp

def learn_probabilities_finite(mdp_object, iteration_num):

    # The structure of MDP is known
    A = mdp_object.actions
    S = mdp_object.states
    P = mdp_object.probabilities # Probabilities needed for transitioning from one state to another

    # Needed for choosing actions that should be executed
    state_actions = {s : 
                        {"iteration_num" : 0, 
                         "actions_num" : None, 
                         "actions" : None} for s in S}
 
    # Create dictionary that stores number of possible actions and possible actions in an array
    for state in S:
        state_actions[state]["actions_num"] = len(list(P[state].keys()))
        state_actions[state]["actions"] = list(P[state].keys())
        
    # Store how many times a transition to a certain state took place
    states_hits = {s : { a : {s : 0 for s in S} for a in A } for s in S}

    prob_denominator = len(S) # Denominator for calculating probabilities

    # Initially all probabilities for transitioning to other state are equal
    initial_prob = 1/prob_denominator
    approximated_prob = copy.deepcopy(P)

    # Set all transition probabilities to initial probability
    for current_state in approximated_prob:
        for action in approximated_prob[current_state]:
            for following_state in approximated_prob[current_state][action]:
                approximated_prob[current_state][action][following_state] = initial_prob

    current_state = 's0' # Start at state 's0'
    i = 0 # Number of iterations

    for _ in range(iteration_num):
        i += 1 # For testing cases

        # Retreive which action should be executed as a next one
        action_to_execute_index = state_actions[current_state]["iteration_num"] % state_actions[current_state]["actions_num"]
        executed_action = state_actions[current_state]["actions"][action_to_execute_index]

        state_actions[current_state]["iteration_num"]+=1 # Increase the iteration number of the current state

        # Get probabilities of transitioning to other states (needed for transition execution)
        states_prob = mdp.get_foll_states_prob_values(P, current_state, executed_action)
        next_state = random.choices(S, weights = states_prob)[0]

        # Mark state to which transition took place
        states_hits[current_state][executed_action][next_state]+=1
    
        # Calculate new approximation of probabilities
        hits_list = list(states_hits[current_state][executed_action].values())  
        hits_num = sum(hits_list) # Sum how many states changes took place for a certain state after executing a certain action
        
        for state in S:
            approximated_prob[current_state][executed_action][state] = (1 + states_hits[current_state][executed_action][state]) / (prob_denominator + hits_num)
    
        current_state = next_state



    # Manual tests
        
    ## 1: the number of all iterations, sum of iterations for each of states and sum of states' hits should be equal
    # states_hits_sum = 0
    # for state, action_foll_state in states_hits.items():
    #     for action, foll_states in action_foll_state.items():
    #         for foll_state, hit in foll_states.items():
    #             states_hits_sum += hit

    # states_iteration_sum = 0
    # for state in S:
    #     states_iteration_sum += state_actions[state]["iteration_num"]

    # print("Equality test: iteration_num === states_hits == states_iteration:", i == states_hits_sum == states_iteration_sum)

    return approximated_prob

def learn_probabilities_convergence(mdp_object, threshold):

    # The structure of MDP is known
    A = mdp_object.actions
    S = mdp_object.states
    P = mdp_object.probabilities # Probabilities needed for changing between transitions

    # Needed for choosing actions that should be executed
    state_actions = {s : 
                        {"iteration_num" : 0, 
                         "actions_num" : None, 
                         "actions" : None} for s in S}
 
    # Create dictionary that stores number of possible actions and possible actions in an array
    for state in S:
        state_actions[state]["actions_num"] = len(list(P[state].keys()))
        state_actions[state]["actions"] = list(P[state].keys())

    # for state in S:
    #     print(state)
    #     print("actions_num:", state_actions[state]["actions_num"])
    #     print("actions:", state_actions[state]["actions"])
    #     print("\n")
        
    # Store how many times a transition to a certain state took place
    states_hits = {s : { a : {s : 0 for s in S} for a in A } for s in S}

    prob_denominator = len(S) # Denominator for calculating probabilities

    # Initially all probabilities for transitioning to other state are equal
    initial_prob = 1/prob_denominator
    approximated_prob = copy.deepcopy(P)
    approximated_prob_new = copy.deepcopy(P)

    # Set all transition probabilities to initial probability
    for state in approximated_prob:
        for action in approximated_prob[state]:
            for next_state in approximated_prob[state][action]:
                approximated_prob[state][action][next_state] = initial_prob
                approximated_prob_new[state][action][next_state] = initial_prob

    current_state = 's0' # Start at state 's0'
    i = 0 # Number of iterations

    # List with possible actions from each state (needed for convergence)
    exe_actions = []
    for state in S: 
        exe_actions.append(state_actions[state]["actions"])
    # print(exe_actions)

    while True:
        i += 1 # For testing cases

        # Retreive which action should be executed as a next one
        action_to_execute_index = state_actions[current_state]["iteration_num"] % state_actions[current_state]["actions_num"]
        executed_action = state_actions[current_state]["actions"][action_to_execute_index]

        state_actions[current_state]["iteration_num"]+=1 # Increase the iteration number of the current state

        # Get probabilities of transitioning to other states (needed for transition execution)
        states_prob = mdp.get_foll_states_prob_values(P, current_state, executed_action)
        next_state = random.choices(S, weights = states_prob)[0]

        # Mark state to which transition took place
        states_hits[current_state][executed_action][next_state]+=1
    
        # Calculate new approximation of probabilities
        hits_list = list(states_hits[current_state][executed_action].values())  
        hits_num = sum(hits_list) # Sum how many states changes took place for a certain state after executing a certain action
        
        for state in S:
            approximated_prob_new[current_state][executed_action][state] = (1 + states_hits[current_state][executed_action][state]) / (prob_denominator + hits_num)

        # Check convergence
        convergence = True

        for (init_state, exe_action_list) in zip(S, exe_actions):
            for exe_action in exe_action_list:
                for following_state in S:
                    convergence = convergence and (abs(approximated_prob[init_state][exe_action][following_state] - approximated_prob_new[init_state][exe_action][following_state]) < threshold)
            
        if convergence:

            # Manual tests

            ## 1: the number of all iterations, sum of iterations for each of states and sum of states' hits should be equal
            # print("Number of iterations: ", i)
            # states_hits_sum = 0
            # for state, action_foll_state in states_hits.items():
            #     for action, foll_states in action_foll_state.items():
            #         for foll_state, hit in foll_states.items():
            #             states_hits_sum += hit

            # states_iteration_sum = 0
            # for state in S:
            #     states_iteration_sum += state_actions[state]["iteration_num"]

            # print("Equality test: iteration_num === states_hits == states_iteration:", i == states_hits_sum == states_iteration_sum)
            return approximated_prob_new
        
        approximated_prob = copy.deepcopy(approximated_prob_new)
        current_state = next_state