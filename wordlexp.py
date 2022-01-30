import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import display
import logomaker
import itertools
import joblib

all_words = joblib.load('all_words.jl')
letter_cols = [f"L-{_}" for _ in range(5)]

class bcolors:
    """
    Color text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def letter_summary(df):
    """
    percent of words each remaining letter is within
    """
    table = df.melt(id_vars=['word'], value_vars=letter_cols)
    table = table.sort_values(['word', 'variable'])
    table = table.drop_duplicates(subset=['word', 'value'])
    table = table.groupby(['value']).size()
    table = table.to_frame('count')
    table['freq'] = table['count'] / len(df)
    table = table.sort_values(['freq'], ascending=False)
    return table
    return overall_freq.to_frame('freq')


def reduce_space(search_space, placed_letters=None, unplaced_letters=None, absent_letters=None):
    """
    Given a set of known information, reduce the search space of round
    """
    if not placed_letters:
        placed_letters = [None, None, None, None]
    if not unplaced_letters:
        unplaced_letters = defaultdict(set)
    if not absent_letters:
        absent_letters = set()
    # For reducing the space - don't need to know answer
    filt = np.zeros(len(search_space))
    for pos, i in enumerate(placed_letters):
        if i is not None:
            filt += search_space[f"L-{pos}"] != i

    for letter, poss in unplaced_letters.items():
        filt += ~search_space[letter_cols].isin([letter]).any(axis=1)
        for pos in poss:
            filt += search_space[pos] == letter

    filt += search_space[letter_cols].isin(absent_letters).any(axis=1)
    return filt


def make_guess(search_space, guess, answer):
    """
    Given a guess and the answer, build a filter of now impossible words
    """
    guess = guess.upper()
    answer = answer.upper()
    placed_letters = [None, None, None, None, None]
    unplaced_letters = defaultdict(set)
    absent_letters = set()
    for pos, ls in enumerate(zip(guess, answer)):
        g, a = ls
        if g == a:
            placed_letters[pos] = g
        elif g in answer:
            unplaced_letters[g].add(f"L-{pos}")
        else:
            absent_letters.add(g)
    filt = reduce_space(search_space, placed_letters,
                        unplaced_letters, absent_letters)
    return filt


def random_guesser(possible_words, answer, chance=0):
    """
    Bot that employs random guessing strategy
    Returns list of [(size, guess)]
    """
    filt = np.zeros(len(possible_words))
    solved = False
    ret = []
    while (filt == 0).sum() > 1 and chance < 6:
        chance += 1
        guess = possible_words[filt == 0]['word'].sample(1).iloc[0]
        n_filt = make_guess(possible_words[filt == 0], guess, answer)
        
        if guess == answer:  # solved
            ret.append([1, 1, chance, True])
            return ret
        filt += n_filt
        a_space = (filt == 0).sum()
        v_space = ((filt == 0) & (possible_words['valid'])).sum()
        ret.append([a_space, v_space, chance, False])
    return ret


def get_random_word():
    """
    Return a new random word
    """
    return all_words[all_words["valid"]]['word'].sample(1).iloc[0]


def make_logo(valid, min_freq=0.10):
    """
    Visualize the word space
    """
    total = len(valid)
    rows = []
    for i in range(5):
        m_row = valid[f"L-{i}"].value_counts() / total
        m_row = m_row.fillna(0)
        m_row[m_row < min_freq] = 0
        rows.append(m_row)
        # all_words[all_words["valid"]][letter_cols].value_counts().reset_index().head()
    seqlogo = pd.concat(rows, axis=1).T
    seqlogo['pos'] = range(5)
    seqlogo.reset_index(drop=True, inplace=True)
    seqlogo.set_index('pos', inplace=True)

    crp_logo = logomaker.Logo(seqlogo.fillna(0),
                              shade_below=.5,
                              fade_below=.5,
                              font_name='Arial Rounded MT Bold')

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    crp_logo.ax.set_ylabel("Letter Frequency", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.xaxis.set_tick_params(pad=-1)
    plt.show()


def print_history(history, answer, summary):
    """
    Semi-wordle coloring of the guess history
    """
    for guess in history:
        ans = list(answer)
        out_str = [''] * len(guess)
        for pos in range(len(guess)):
            if guess[pos] == answer[pos]:
                out_str[pos] = bcolors.BOLD + \
                    bcolors.OKGREEN + guess[pos] + bcolors.ENDC
                ans.remove(guess[pos])
        for pos in range(len(guess)):
            if out_str[pos] != '':
                continue
            if guess[pos] in ans:
                out_str[pos] = bcolors.BOLD + \
                    bcolors.WARNING + guess[pos] + bcolors.ENDC
            else:
                out_str[pos] = guess[pos].lower()
        print("".join(out_str))
    print(summary)


def look_at_remaining(filt, guess):
    """
    Get a peek at the remaining words
    """
    if len(guess) > 1:
        head = int(guess[1:])
    else:
        head = 5
    valid = all_words[(filt == 0) & all_words['valid']]
    head = min(head, (filt == 0).sum())
    display(all_words[filt == 0].sample(head))
    print('space', (filt == 0).sum(), '(', len(valid), ')')


def look_at_valid_remaining(filt, guess):
    """
    Look at remaining valid wors
    """
    if len(guess) > 1:
        head = int(guess[1:])
    else:
        head = 5
    valid = all_words[(filt == 0) & all_words['valid']]
    head = min(head, len(valid))
    display(valid.sample(head))
    print('space', (filt == 0).sum(), '(', len(valid), ')')


def available_letters(history):
    """
    Prints the remaining letters available
    """
    s = set(itertools.chain.from_iterable(history))
    remain = set([chr(_) for _ in range(ord('A'), ord('Z') + 1)]) - s
    print(" ".join(remain))


def play(answer=None):
    """
    Play a round of wordle (unlimited guesses)
    """
    if not answer:
        answer = get_random_word()
    answer = answer.upper()
    history = []
    guess = ""
    filt = np.zeros(len(all_words))
    summary = ""
    while not guess.startswith('$'):
        guess = input('guess: ').upper()
        if guess.startswith('#'):
            look_at_remaining(filt, guess)
            continue

        elif guess.startswith('^'):
            look_at_valid_remaining(filt, guess)
            continue

        elif guess.startswith(';'):
            available_letters(history)
            continue

        elif guess.startswith('@'):
            # Turn this into a method. Also, don't double color? Also, fix colors
            print_history(history, answer, summary)
            continue

        # Get letter summary of remaining words
        if guess.startswith('%'):
            lfreq = letter_summary(all_words[filt == 0])
            display(lfreq[lfreq['freq'] != 1].head(5).T.round(2))
            make_logo(all_words[filt == 0])
            continue

        # Reset filter
        if guess.startswith('_'):
            filt = np.zeros(len(all_words))
            if len(guess) > 1:
                answer = all_words[all_words["valid"]
                                   ]['word'].sample(1).iloc[0]
            history = []
            answer = get_random_word()
            print('reset')
            continue
        # Update filter. Syntax is .<pos><op><letter>
        # where pos is the position 0-4, letter is the letter
        # = - placed letter
        # ? - unplaced letter
        # ! - absent letter
        if guess.startswith("."):
            dat = guess[1:].upper()
            if len(dat) != 3:
                print('invalid format')
                continue
            pos = int(dat[0])
            if dat[1] == '=':
                placed = [None, None, None, None, None]
                placed[pos] = dat[2]
                filt += reduce_space(all_words, placed)
            if dat[1] == '?':
                filt += reduce_space(all_words,
                                     unplaced_letters={dat[2]: [f"L-{pos}"]})
            if dat[1] == '!':
                filt += reduce_space(all_words, absent_letters=[dat[2]])
            continue
        # Invalid guess
        if len(guess) != 5 or ~all_words['word'].isin([guess]).any():
            print("Invalid guess")
            continue

        history.append(guess)
        # Make guess
        filt += make_guess(all_words, guess, answer)
        cnt = (filt == 0).sum()
        tot = len(filt)
        pct = cnt / tot * 100
        valid_cnt = len(all_words[(filt == 0) & (all_words['valid'])])
        summary = f"{cnt} / {tot} remaining {pct:.1f}% - {valid_cnt}"
        print_history(history, answer, summary)
        if guess == answer:
            print("Finished!")
            break
        # Output the colored latest guess


def evaluate_first_word(guess, samples=3, n=None, plot=False):
    """
    Evaluate a first guess by plotting the space size, win distribution, and loss percent
    samples - number of times to try 
    n - number of valid words to guess agains

    Default of samples=1, n=None will try every valid word once
    e.g. samples=2, n=10 will make two attempts through 10 random words
    """
    # Only going to make answers for a sampling of valid words
    if n is None:
        n = all_words['valid'].sum()
    guess = guess.upper()
    results = []
    cnt = []
    for _ in range(samples):
        for answer in all_words[(all_words["valid"])]['word'].sample(n, random_state=42):
            # Make the first guess
            filt = make_guess(all_words, guess, answer)
            won = guess == answer
            a_space = (filt == 0).sum()
            v_space = ((filt == 0) & (all_words['valid'])).sum()
            cnt.append([a_space, v_space,  1, won])
            # Continue randomly guessing from there
            results = random_guesser(all_words[filt == 0], answer, 1)
            cnt.extend(results)
    cnt = pd.DataFrame(cnt, columns=["a_space", "v_space", "guess", 'won'])
    summary = cnt[cnt["won"]].groupby(['guess']).size().to_frame('win count').reset_index()
    total_losses = len(cnt[(~cnt["won"]) & (cnt["guess"] == 6)])
    percent_losses = (total_losses / (n * samples)) * 100
    speed = cnt[cnt['won']]['guess'].describe()

    if plot:
        p = sb.boxplot(data=cnt, x="guess", y="a_space")
        p.set(title="Word Space", xlabel="Guess Number",
              ylabel="Space Size", yscale='log')
        p.set_yticklabels(["{:,}".format(int(_)) for _ in p.get_yticks()])
        plt.grid(which='major', axis='both')
        sb.despine()
        plt.show()
        display(cnt.groupby('guess')['a_space'].median().T)

        p = sb.boxplot(data=cnt, x="guess", y="v_space")
        p.set(title="Valid Word Space", xlabel="Guess Number",
              ylabel="Space Size", yscale='log')
        p.set_yticklabels(["{:,}".format(int(_)) for _ in p.get_yticks()])
        plt.grid(which='major', axis='both')
        sb.despine()
        plt.show()
        display(cnt.groupby('guess')['v_space'].median().T)

        p = sb.barplot(data=summary, x="guess", y="win count")
        p.set(title="But I am bad at it", xlabel="Guess Count", ylabel="Win Count")
        plt.show()
        print('loss:', total_losses, "%.2f%%" % (percent_losses))
        print("speed:", speed)

    return {'word': guess, 'speed': speed['mean'], 'loss': percent_losses}

def test():
    history = ["RINSE",
               "REPAY",
               "ROLES",
               "ROVER",
               "RODEO"]
    print_history(history, "RODEO", '')
if __name__ == '__main__':
    test()
