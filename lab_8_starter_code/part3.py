from gridworld import GridWorld

# instantiate the gridworld object
world = GridWorld(size=5, 
                  state_includes_location=False, 
                  state_includes_sensory=True,  # this agent has features describing the state
                  randomize_each_trial=False
                  )

# Write an agent that uses function approximation to solve this world







# As in task 1, visualize the agent navigating the world after it has finished learning. 
# Why can this agent handle the re-randomizing world better than the Q-learning agent from task 1?




# Why would a Q-learning agent take much longer to learn from these features, than from the 
# state in task 1? Why does function approximation not suffer from the same problem?
