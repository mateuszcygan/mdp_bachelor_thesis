import random

# SETS
def generate_states():

    S = []
    num_states = random.randint(3,9)

    for state_num in range(num_states):
        state = 's' + str(state_num)
        S.append(state)

    return S

def generate_actions():

    A = []
    num_actions = random.randint(2,5)

    for action_num in range(num_actions):
        action = 'a' + str(action_num)
        A.append(action)

    return A

# Rewards
# After executing each action certain reward is assigned to each following state
# current state : { action : { following state : reward, following state : reward ... }}
# R = {
#       's0' : {'a0' : { 's0' : -2, 's1' : 2, 's2' : 5},
#               'a1' : { 's0' : 3, 's1' : -2, 's2' : 5}
#               'a2' : { 's0' : -9, 's1' : 5, 's2' : -5}
#       }
#       's1' : {'a0' : { 's0' : 0, 's1' : 2, 's2' : 8},
#              { 'a1' : { ... }}
#       }
#     }
def generate_rewards(states, actions):
    return {state : {action : {state : random.randint(-5, 5) for state in states} for action in actions} for state in states}

def gen_values_for_prob(states, actions):
    # Number of needed probability values = states * actions * states
    # Number of states coresponds to number of probability values for each action - 'a0' : { 's0' : 0.75, 's1' : 0.2, 's2' : 0.05}
    
    prob_values = []
    
    for state in states:
        for action in actions: # From each state we can activate each action
            total_prob = 0
            prob_val_action = []

            for state in states[:-1]: # Exclude the last state, as it will be adjusted later
                prob = round(random.uniform(0, 1 - total_prob), 2)
                total_prob += prob
                prob_val_action.append(prob)

            # Adjust the probability of the last state to ensure the sum is exactly 1.0
            last_prob = round(1 - total_prob, 2)
            prob_val_action.append(last_prob)
            prob_values.append(prob_val_action)

    num_of_val = len(states) * len(actions) * len(states)

    # print("Correct number of prob_values in array?:", num_of_val == len(prob_values)) # /list have to be flatten
    # print(prob_values)
    return prob_values
    
# Probabilities that action a in state s at time t will lead to state s' at time t+1
# current state : { action : {following state : probability, following state : probability ... }}
# P = {
#       's0' : {'a0' : { 's0' : 0.75, 's1' : 0.2, 's2' : 0.05},
#               'a1' : { 's0' : 0.3, 's1' : 0.2, 's2' : 0.5}
#               'a2' : { 's0' : 0.9, 's1' : 0.05, 's2' : 0.05}
#       }
#       's1' : {'a0' : { 's0' : 0.0, 's1' : 0.2, 's2' : 0.8},
#              { 'a1' : { ... }}
#       }
#     }
def generate_prob(states, actions):

    prob_val = gen_values_for_prob(states, actions)
    prob_dict = {state : { action : {state : None for state in states} for action in actions } for state in states}
    action_number = 0

    for state in prob_dict:
        for action in prob_dict[state]:

            state_with_prob_number = 0

            for state_with_prob in prob_dict[state][action]:

                prob_dict[state][action][state_with_prob] = prob_val[action_number][state_with_prob_number]
                state_with_prob_number += 1

            action_number += 1
    # print(prob_dict)
    return prob_dict

# Going over to another state after selecting the action
def state_transition(P, current_state, chosen_action):
    prob_current_action = list(P[current_state][chosen_action].values())# Get probabilities for going over to another state from the current action
    following_states = list(P[current_state][chosen_action].keys())
    
    print("Probabilities of going to another state after", chosen_action, ":", prob_current_action)
    print("Possible following states:", following_states)

    # Go over to another state based on probability of certain action
    next_state = random.choices(following_states, weights = prob_current_action)[0]
    print("Next state:", next_state)

    # print("Based on probability chosen next state:", next_state, '\n')

    return next_state



# MDP
class MDP:
    def __init__(self, states, actions, rewards, probabilities):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.probabilities = probabilities
    
    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions
    
    def get_rewards(self):
        return self.rewards
    
    def get_probabilities(self):
        return self.probabilities
    
    def get_properties(self):
        return self.states, self.actions, self.rewards, self.probabilities

# Return an array with rewards for going over to following states through an executed action from a current state
def get_foll_states_rewards_values(R, current_state, executed_action): 
    return list(R[current_state][executed_action].values())

# Return an array with probabilities for going over to following states through an executed action from a current state
def get_foll_states_prob_values(P, current_state, executed_action):
    return list(P[current_state][executed_action].values())

# Return an array with possible actions that can be executed from a certain state
def get_possible_actions(P, state):
    return list(P[state].keys())

def createMDP():
    # S = ['s0', 's1', 's2'] #state space
    # A = ['a0', 'a1'] #action space
    S = generate_states()
    A = generate_actions()

    R = generate_rewards(S, A)
    P = generate_prob(S, A)

    return MDP(S, A, R, P)



# SPARSE DENSE

# Calculate number of possible states' transitions
def calculate_num_of_states_transitions(mdp_object):

    P = mdp_object.probabilities
    S = mdp_object.states
    num_of_states = len(S)

    poss_states_transitions = 0 # Includes also states with transition's probability equal 0

    for state in S:
        executable_actions = get_possible_actions(P, state)
        num_of_actions = len(executable_actions)
        poss_states_transitions += num_of_states * num_of_actions

    return poss_states_transitions

# Function that outputs the number of states with a probability of 0.0 (indicating unreachable states)
def unreachable_states(mdp_object):

    states = mdp_object.states
    actions = mdp_object.actions
    P = mdp_object.probabilities

    poss_states_transitions = calculate_num_of_states_transitions(mdp_object)

    unreachable_overall = 0

    for state in states:
        executable_actions = get_possible_actions(P, state)
        for action in executable_actions:
            prob_values = list(P[state][action].values())
            unreachable = 0
            for prob_value in prob_values:
                if prob_value == 0.0:
                    unreachable += 1
            # print("Unreachable states:", unreachable)
            unreachable_overall += unreachable

    print("tran_to_other_states(incl prob=0):", poss_states_transitions)
    print("unreach_states(prob=0):", unreachable_overall)
    
    unreachable_percentage = unreachable_overall/poss_states_transitions
    print("percent_of_unreach_states:", unreachable_percentage)
    print("\n")

# Function that outputs the number of following states with a rewards of 0.0
def zero_rewards(mdp_object):

    states = mdp_object.states
    actions = mdp_object.actions
    P = mdp_object.probabilities
    R = mdp_object.rewards

    rewards_num = len(states)*len(states)*len(actions)

    zero_overall = 0

    for s in states:
        executable_actions = get_possible_actions(P, s)
        for a in executable_actions:
            rewards = list(R[s][a].values())
            zero = 0
            for reward in rewards:
                if reward == 0:
                    zero += 1
            # print("Unreachable states:", unreachable)
            zero_overall += zero

    print("num_all_rewards:", rewards_num)
    print("zero_rewards(r=0):", zero_overall)
    
    zero_percentage = zero_overall/rewards_num
    print("percent_of_zero_rewards:", zero_percentage)

# For states without following actions (all probabilities are set to zero) randomly choose one following state
def random_following_state(mdp):

    S = mdp.states
    A = mdp.actions
    P = mdp.probabilities

    for current_state in S:
        executable_actions = get_possible_actions(P, current_state)
        for action in executable_actions:
            transition_prob = list(P[current_state][action].values())

            # If all probabilities are zero (no following state), choose randomly one action and assign probability 1.0 to it
            if sum(transition_prob) == 0:
                random_state = random.choice(S)
                P[current_state][action][random_state] = 1.0

# Normalizing probabilities for following states of a certain MDP
def normalize_mdp_probabilities(mdp):

    S = mdp.states
    A = mdp.actions
    P = mdp.probabilities

    for current_state in S:
        executable_actions = get_possible_actions(P, current_state)
        for action in executable_actions:

            # Calculate the total value of probabilities for certain action
            action_prob = list(P[current_state][action].values())
            total = sum(action_prob)

            for foll_state in S:

                # Normalize the probability of following states
                try:
                    P[current_state][action][foll_state] = P[current_state][action][foll_state]/total
                except ZeroDivisionError:
                    P[current_state][action][foll_state] = 0

# Thinning the reachable states in the MDP by using state_sparsity_rate - percentage of states that should be unreached
def sparse_mdp_states(mdp, state_sparsity_rate):

    S = mdp.states
    A = mdp.actions
    P = mdp.probabilities
    
    old_probability_weight = 1 - state_sparsity_rate
    sparsity_weights = [state_sparsity_rate, old_probability_weight] # Sparse weights for 0.0 probability and probability of going over to following state

    for current_state in S:
        executable_actions = get_possible_actions(P, current_state)
        for action in executable_actions:
            for following_state in S:

                possible_prob_values = [0.0]  # Array containing 0.0 probability and probability of transitioning to following state (used for decision-making)
                prob_value = P[current_state][action][following_state]

                # If any of the following states is already unreachable, don't change it
                if prob_value == 0:
                    continue

                possible_prob_values.append(prob_value)

                # Choose between 0.0 and old probability of transitioning to following state
                new_prob_value = random.choices(possible_prob_values, weights = sparsity_weights)[0]
                P[current_state][action][following_state] = new_prob_value

    normalize_mdp_probabilities(mdp)
    random_following_state(mdp)

def sparse_mdp_rewards(mdp, reward_sparsity_rate):

    S = mdp.states
    A = mdp.actions
    R = mdp.rewards
    P = mdp.probabilities

    old_probability_weight = 1 - reward_sparsity_rate
    sparsity_weights = [reward_sparsity_rate, old_probability_weight] # Sparse weights for 0.0 reward and assigned reward

    for current_state in S:
        executable_actions = get_possible_actions(P, current_state)
        for action in executable_actions:
            for following_state in S:

                # possible_reward_values = [0]
                possible_reward_values = [0.0]  # Array containing 0.0 reward and assigned reward (used for decision-making) 
                reward_value = R[current_state][action][following_state]

                # If any of the rewards is already equal 0, don't change it
                if reward_value == 0:
                    continue

                possible_reward_values.append(reward_value)

                # Choose between reward 0.0 and assigned reward
                new_reward_value = random.choices(possible_reward_values, weights = sparsity_weights)[0]
                R[current_state][action][following_state] = new_reward_value

def reduce_actions_number(mdp, min_num=1, max_num=None):

    """
    Reduce the number of possible actions for each state

    Args:
    - mdp: An object representing the MDP.
    - min_num: The minimum number of possible actions from a state. Defaults to 1.
    - max_num: The maximum number of possible actions from a state. If not specified, defaults to the total number of actions available in the MDP.

    Returns:

    Test: executable actions in rewards and probabilities for the same states are the same
    """

    # If max_num not defined, assign the value of possible states
    if max_num is None:
        max_num = len(mdp.actions)
    
    A = mdp.actions
    S = mdp.states
    R = mdp.rewards
    P = mdp.probabilities
    
    for state in S:

        actions_num = random.randint(min_num, max_num) # Randomly choose a number from a given interval (number of actions that should not be deleted)
        random_actions = random.sample(A, k=actions_num) # Randomly choose concrete actions that should not be deleted

        actions_to_delete = set(A) - set(random_actions) # Determine actions that should be deleted for certain state (the remaining actions)

        for action_to_delete in actions_to_delete:
            del R[state][action_to_delete]
            del P[state][action_to_delete]

    return mdp



# PRINT
def print_mdp_details(dict):
    for state, action_dict in dict.items():
        print(state)
        for action, prob in action_dict.items():
            print(action, ":", prob)
        print('\n')