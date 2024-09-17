# Markov Decision Process Learning and Strategy Optimization

This repository contains the code and methodologies developed for my Bachelor's thesis, which focuses on solving an initially unknown Markov Decision Process (MDP) by learning its structure and computing an optimal strategy within a limited time frame.

## Algorithm structure
Two algorithms were developed, both of which adhere to the constraints of limited time. Each algorithm consists of two distinct phases, which are based on the same underlying concepts:

### Exploration Phase
In this phase, the algorithm systematically explores the model to learn its structure. A modified version of Dijkstra’s algorithm is employed to attempt reaching the least-visited states, ensuring a thorough exploration of the MDP.

### Exploitation Phase
After the exploration phase, the Value Iteration algorithm is used as the basis for this phase. It leverages the approximated transition probabilities and rewards to compute an optimal policy for the MDP.

## Algorithm difference
Although both algorithms share the same core structure, they differ in how they determine when to transition from the exploration (learning) phase to the exploitation (optimization) phase:

### First Algorithm - Iteration-Based Transition
In the first algorithm, the user can specify the number of iterations to be spent in the learning phase. Once this predefined number of iterations is completed, the algorithm switches to the exploitation phase.

### Second Algorithm - Knowledge-Based Transition
The second algorithm allows the user to specify a threshold for how well the MDP should be understood before leaving the learning phase. Instead of a fixed number of iterations, the algorithm evaluates how well it has approximated the transition probabilities and rewards, and only proceeds to the exploitation phase when the specified level of knowledge is achieved.
