# Wordle Explorer

A reimplementation of Wordle built for exploratory data analysis.


# Motivation

Everyone is going crazy over Wordle. 
There have been lots of
[conversation](https://www.gamespot.com/articles/the-best-wordle-starting-word-has-been-figured-out-with-computer-science/1100-6500073/)
about what's the best starting word. Two suggestions I've seen are `irate` (proposed by a linguist) and `later` 
(proposed by a computer scientist).

As a bioinformatician, I thought it'd be fun to dig in and use my data-science training to find the best starting word.

Like most projects, this turned into a much bigger thing that I imagined, and there are interesting patterns that I'll
continue to explore. But, let's get to the answer for now.

# What is the 'best word'
One thing that's bugged me about these best word conversations is what's 'BEST' hasn't been defined. The way I see it,
there's two ways a starting word can be good.

1. Speed - The number of guesses, on average, that a starting word takes to get to the answer
2. Loss - The number of rounds, on average, that a starting word does not find the answer

# On Average?
Yes. I've made a 'random' guesser that will play Wordle for us. This bot is probably above average in skill because it
knows every possible 5 letter answer word and plays Wordle's hard mode perfectly. It looks at the space of valid,
possible words and randomly picks one to be its next guess.

# Sure. Average enough. So what's the best word!?

Here are the top 5 fastest words.
| word  | speed |
|-------|-------|
| STEAL | 4.164 |
| PLATE | 4.165 |
| BLAST | 4.166 |
| SLATE | 4.166 |
| PLANT | 4.171 |

The fastest word seems to be `steal`, which wins in about 4.1 guesses. We can see that `later` is a fairly 
bove average word. But there are faster words.
![](imgs/speed_later.png?raw=true)

Here are the top 5 winningest words:
| word  | loss  |
|-------|-------|
| SLEPT | 7.156 |
| SPLAT | 7.271 |
| PLANT | 7.372 |
| STOMP | 7.372 |
| CLAMP | 7.401 |

Our least likely to lose word is `slept`, which lost only 7.1% of the time. We can see that `later` is below
average.
![](imgs/loss_later.png?raw=true)

And if you like to take a balanced approach to Wordle. The overall best words are:
| word  | score |
|-------|-------|
| SLEPT | 5.679 |
| SPLAT | 5.735 |
| PLANT | 5.771 |
| STOMP | 5.826 |
| CLAMP | 5.835 |

# SLEPT wins!

But!, it won because our random guesser bot knows all the words. You can practice your wordle-ing to
learn more words and maybe develop a strategy better than random guessing with `Wordle.ipynb`. 
As you play, you can quickly get hits of possible guesses, tables of remaining letter frequencies, 
and plots of positional information. 

![](imgs/game.png?raw=true)

# Requirements

- numpy
- pandas
- seaborn
- logomaker
- joblib
