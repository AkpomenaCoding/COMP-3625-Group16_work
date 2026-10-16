from random import choice, randint, random
from translator import UniversalTranslator
from matplotlib import pyplot as plt
import numpy as np

"""
    function name: decorate rate 
    params: translated_string
    description: takes the string that has been translated and returns the percentage of that string are words
                words are seen the success in the string.
"""
def decode_rate(translated_string):
    tokens = translated_string.split()
    words = [t for t in tokens if not t.isdigit()]

    rate = len(words)/len(tokens)
    
    return rate

"""
    function name: plot_decode_rate
    params: settings, decode_rate
    description: 
        plots the graph of settings against their decode rates 
        Adapted from Eric's code from the assignment google doc
"""

# given settings: a Nx2 array of N setting combinations
# and decode_rate: a length-N array of decode rates corresponding to those settings

def plot_decode_rate(settings, decode_rate):
    # generate a scatter plot
    plt.scatter(x=settings[:, 0],
            y=settings[:, 1],
            c=decode_rate,
            vmin=0, vmax=1
            )

# add colorbar and gridlines
    cbar = plt.colorbar(label="decode rate")
    plt.grid()

# add labels
    plt.xlabel('knob 0 setting')
    plt.ylabel('knob 1 setting')
    plt.title('decode rates for settings tried')

# display
    plt.show()

"""
    function name: decode_hill_climb
    params: 
    description: 
        decode rate: words translated / total words in the message
        hill-climbing for the maximum decode rate and minimizing finding optimal settings 
        this will be done through starting each knob at a random value, 
        and then iteratively adjusting each knob to find the best setting for that knob, 
        while keeping the other knobs fixed. This will be repeated until no further improvements can be made
"""
def decode_hill_climb(translator: UniversalTranslator, num_knobs: int, num_steps: int, improvement_judge: int) -> tuple[list, float]:
    scores = []
    all_settings = [] 

    best_settings = [round(random(), 1) for setting in range(num_knobs)]
    best_trans_string = translator.translate(best_settings)
    best_dec_rate = decode_rate(best_trans_string)
    scores.append(best_dec_rate)
    all_settings.append(best_settings)
    
    steps_without_improvement = 0
    for step in range(num_steps):
        curr_settings = best_settings.copy()
        knob = randint(0, num_knobs - 1)
        curr_settings[knob] += choice([-0.1, 0.1])
        curr_settings[knob] = round(curr_settings[knob], 2)
        curr_settings[knob] = max(0.0, min(1.0, curr_settings[knob]))
        
        curr_trans_string = translator.translate(curr_settings)
        curr_dec_rate = decode_rate(curr_trans_string)
        scores.append(curr_dec_rate)
        all_settings.append(curr_settings)

        if curr_dec_rate > best_dec_rate:
            best_settings = curr_settings
            best_dec_rate = curr_dec_rate
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        print(f'Step {step + 1}: Current settings: {curr_settings}, decode rate: {curr_dec_rate:.0%}')
        
        # if the algorithm isn't finding a good rate to decode within 30 steps then it stops, saving us time
        if steps_without_improvement >= improvement_judge: 
            break

    return best_settings, best_dec_rate,  all_settings, scores


if __name__ == "__main__":
    # create the UniversalTranslator object, with 2 knobs
    translator = UniversalTranslator(n_dim=2)

    best_settings, best_dec_rate,  all_settings, scores = decode_hill_climb(translator, num_knobs=2, num_steps=50, improvement_judge=30)

    # print the reults
    print(f'Final translation: {translator.translate(best_settings)}')
    print(f'Final result: {best_settings}, decode rate: {best_dec_rate:.0%}')

    # print total number of settings evaluated
    print(f'# settings tried: {translator.n_settings_tried()}')

    plot_decode_rate(np.array(all_settings), scores) 