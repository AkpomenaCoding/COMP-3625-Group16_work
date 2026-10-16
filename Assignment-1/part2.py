import math
import time
from random import choice,randint, random
from translator import UniversalTranslator
from part1 import decode_rate

def cost(settings):
    return 1 - decode_rate(translator.translate(settings))

def simulated_annealing(f, start, n, T0, step): # f is a cost
    current_settings = start.copy()
    best_settings = current_settings.copy()
    current_cost = f(current_settings)
    best_cost = current_cost
    
    alpha = 0.9995
    step_max = 0.1
    step_min = 0.01

    for i in range(n) :
        T = T0 * alpha**i # geometric approach to doing this
        if T <= 0:
            break
        #step = (step_max - step_min) * (1 - i/n) + step_min
        next_settings = current_settings.copy()
        k = randint(0, len(next_settings) - 1) # my random successor
        next_settings[k] = round(max(0.0, min(1.0, next_settings[k] + choice([-step, step]))), 2)

        next_cost = f(next_settings)
        dE = next_cost - current_cost  # dE is delta E
        
        if dE < 0 or random() < math.exp(-dE/T):
            current_settings = next_settings
            current_cost = next_cost
            
            if current_cost < best_cost:
                best_settings = current_settings
                best_cost = current_cost

    return best_settings

if __name__ == "__main__":

    # create the UniversalTranslator object, with 10 knobs
    translator = UniversalTranslator(n_dim=10)

    start = [round(random(), 2) for knob in range(10)]
    best = simulated_annealing(cost, start, n=50000, T0=0.1, step=0.05)

    print(f'Final translation: {translator.translate(best)}')
    print(f'Decode rate: {1 - cost(best):.4%}')
    # print total number of settings evaluated
    print(f'# settings tried: {translator.n_settings_tried()}')