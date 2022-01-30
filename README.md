# Wordle Explorer

A reimplementation of Wordle built for exploratory data analysis.


# Motivation

Everyone is going crazy over Wordle. 
There have been lots of
[conversations](https://www.gamespot.com/articles/the-best-wordle-starting-word-has-been-figured-out-with-computer-science/1100-6500073/)
about what's the best starting word. Two suggestions I've seen are `irate` (proposed by a linguist) and `later` 
(proposed by a computer scientist).

As a bioinformatician, I thought it'd be fun to dig in and use my data-science training to find the best starting word.

But after my whole Saturday was spent looking at Digrams, and PCAs, and logo plots, I got tired and decided to just brute force the search.

# What is the 'best word'?
One thing that's bugged me about these best word conversations is what's 'BEST' hasn't been defined. The way I see it,
there's two ways a starting word can be good.

1. Speed - The number of guesses, ON AVERAGE, that a starting word takes to get to the answer
2. Loss - The number of rounds, ON AVERAGE, that a starting word does not find the answer

# On Average?
I've made a 'random' guesser that will play Wordle for us. This bot is probably above average in skill because it
knows every possible 5 letter answer word and plays Wordle's hard mode perfectly. It looks at the space of valid,
possible words and randomly picks one to be its next guess. To get the tables below, I had the bot use every valid word 
as it's first guess and then play with every valid word as the answer three times. That comes out to ~7.5 million games 
of Wordle, which took about 5 hours on 16 cores. 

# Sure. Average enough. So what's the best word!?

Here are the top 5 fastest words.
| word  | speed |
|-------|-------|
| STEAL | 4.164 |
| PLATE | 4.165 |
| BLAST | 4.166 |
| SLATE | 4.166 |
| PLANT | 4.171 |

The fastest word seems to be `steal`, which wins in about 4.1 guesses. 
We can see that `later` is a fairly good (red line), but there are faster words.

![](imgs/speed_later.png?raw=true)

Here are the top 5 winningest words:
| word  | loss  |
|-------|-------|
| SLEPT | 7.156 |
| SPLAT | 7.271 |
| PLANT | 7.372 |
| STOMP | 7.372 |
| CLAMP | 7.401 |

Our least likely to lose word is `slept`, which lost only 7.1% of the time. We can see that `later` is average.

![](imgs/loss_later.png?raw=true)

If you like to take a balanced approach to Wordle. The overall best words are (lower score is better):
| word  | score  |
|-------|--------|
| PLANT | -2.235 |
| BLAST | -2.105 |
| SLEPT | -2.094 |
| SPLAT | -2.064 |
| SPENT | -1.955 |

# 🌱 wins!

But!, it won because our bot knows all the words. You can practice your Wordle-ing to
learn more words and maybe develop a strategy better than random guessing with `Wordle.ipynb`. 
As you play, you can quickly get lists of possible guesses, tables of remaining letter frequencies, 
and plots of positional information. 

![](imgs/game.png?raw=true)

# Requirements

- numpy
- pandas
- seaborn
- logomaker
- joblib
- scikit-learn

# Code

Run `build_word_df.py` to start.  
See `wordlexp.py` for details on methods.  
Brute force search done with `score_words.py`.  
Code is very messy.
