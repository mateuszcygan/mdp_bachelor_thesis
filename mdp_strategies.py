import math

import algorithm
import mdp
import policies


# after a certain number of iterations, value iteration is calculated and agent starts to follow calculated strategy
def iterations_number_approach(
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
