import math

import algorithm
import mdp
import policies


# finds the biggest and the smallest reward within the 'learned_rewards' dictionary
def find_rewards_extremes(learned_rewards):

    rewards_extremes = {"min_reward": 0, "max_reward": 0}

    find_rewards_extremes_recursive(learned_rewards, rewards_extremes)

    return rewards_extremes


def find_rewards_extremes_recursive(learned_rewards, rewards_extremes):

    for key, value in learned_rewards.items():
        if isinstance(value, dict):
            find_rewards_extremes_recursive(value, rewards_extremes)
        else:
            if value > rewards_extremes["max_reward"]:
                rewards_extremes["max_reward"] = value
            elif value < rewards_extremes["min_reward"]:
                rewards_extremes["min_reward"] = value


# after a certain number of iterations, value iteration is calculated and agent starts to follow calculated strategy
def iterations_num_strategy(
    # needed for 'iterations_number_approach'
    mdp_object,
    overall_iterations_num,
    # needed for my_algo_alternating
    my_algo_alter_percentage,  # percentage of overall iterations that should be performed in 'my_algo_alternating'
    sys_learn_iterations,
    dijkstra_iterations,
    desired_states_hits_update_percentage,
    total_desired_states_hits_num,
    total_threshold,
    # needed for value_iteration
    value_iteration_threshold,
    value_iteration_dis_factor,
):

    # Calculations for case when when the number of iterations in different approaches can't be exactly estimated (doubles)
    my_algo_iterations = overall_iterations_num * my_algo_alter_percentage
    outer_iterations = math.floor(
        (my_algo_iterations / (sys_learn_iterations + dijkstra_iterations)) * 2
    )

    approximated_prob, learned_rewards, rewards_sum, states_hits = (
        algorithm.my_algo_alternating(
            mdp_object,
            # number of alternating iterations
            outer_iterations,
            # "optional" values
            desired_states_hits_update_percentage,
            total_desired_states_hits_num,
            total_threshold,
            # number of iterations
            sys_learn_iterations,
            dijkstra_iterations,
        )
    )

    V, policy = policies.value_iteration(
        mdp_object, value_iteration_threshold, value_iteration_dis_factor
    )
