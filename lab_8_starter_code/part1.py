from gridworld import GridWorld

# instantiate the gridworld object
world = GridWorld(size=5, 
                  state_includes_location=True, 
                  state_includes_sensory=False, 
                  randomize_each_trial=False
                  )


# do Q learning here
# plot the cumulative reward vs. step #
# and after learning, visualize the agent's behavior in the world




# Now set the world to re-randomize after each trial. What goes wrong? Why?