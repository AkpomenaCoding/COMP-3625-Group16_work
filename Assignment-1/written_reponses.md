Write your answers to the following questions in this file, *after* completing everything else.

# 1) How many settings must be evaluated to *exhaustively* search for the best set (e.g. through some kind of brute-force search). State your assumptions and explain how you arrived at your answer. How does your solution compare to this (quantitatively)?
Brute force could take a lot of time and, it depends on the step size and the number of knobs. Let's say the step size used was 0.1 then from 0, 0.1, 0.2, ..., 1, that's 11 numbers to get a result from that it could around 11 x 11 which is 121 in its worst case. If the step size was 0.01, this could go from 121 to 10201. For the part 2, it could be way more with 10 knobs, 11^10 and that's about 26 billion settings. Our solution solved most the problem for part 1 with around 40 settings that's about less than half the settings needed used. Additionally in part 2, our solution completes about 40% with 50000 settings.




# 2) Why is the 10-knob problem harder than the 2-knob problem? (It may help you to know that the *mechanics* of both problems were the same: the relationship between settings and performance had the same general characteristics, there were just more settings to tune in the 10-knob problem).
The transition from searching different combinations 2 dimensions to 10 dimensions causes a drastic change. Despite the number being between 0 and 1, including the decimal numbers, this allows for many more combinations to exist. Introducing 10 dimensions drastically increases the search and exploration. The combinations increase drastically and exploration is much more needed as one change could ruin the whole combination or improve it altogether.




# 3) If your program was seen as an "agent program", which of the agent types discussed in class would it be?
It would be a utility-based agent due to the nature of it looking for the best optimal settings to translate within a given amount of time. Although the goal is to ideallhy fully translate the transmission with time being short what is the best option through testing and exploration gives it the nature of being utility based.




# 4) it's often said that the simplest solution is the best. How well would a basic hill-climbing search perform in this problem? Justify your answer using your findings or plots from part 1 (you can answer this question whether or not you used hill-climbing as your approach). 
Hill-climbing performs decently well. Although it does have chances of getting stuck in a locla optima, which makes one have to bring in solutions to avoid that. But when transitioning to the 10 dimension aspect of the problem, hill-climbing was not particularly great in terms of exploring further. It had more of a light tread in exploration but was more on dialing things down to find the best solution. This would eventually either lead to the local optima or maximal optima.




# 5) When you moved from the 2- to the 10-knob problem, did you change your search algorithm? Why or why not?
The algorithm had to change from hill climbing to simulated annealing. Thsi decision came about realizing that the first algorithm got stuck in a local optima in the 0 and decimal percentage rising occasionally in the 40%. Due to the expansion from 2d to 10d the hill climbing wasn't searching through most knobs and would get stuck optimizing little rather than exploring to see if one solution could beat the other.