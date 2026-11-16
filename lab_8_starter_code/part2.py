from gridworld import GridWorld

# instantiate the gridworld object
world = GridWorld(size=5, 
                  state_includes_location=True, 
                  state_includes_sensory=False, 
                  randomize_each_trial=False
                  )


# do model-based reinforcement learning here.
# you could recycle the value iteration part from the previous lab.






# Compare this agent’s cumulative reward curve to the Q-learning agent. 
# How much faster does it solve the problem? Explain why in your own words.