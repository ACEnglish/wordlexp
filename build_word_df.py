import pandas as pd
import joblib
# Build words 
valid_words = pd.read_csv("word_list.txt", header=None)
valid_words.columns = ['word']
valid_words["word"] = valid_words["word"].str.upper()
valid_words['valid'] = True
valid_guesses = pd.read_csv("valid_guesses.txt", header=None)
valid_guesses.columns = ['word']
valid_guesses["word"] = valid_guesses["word"].str.upper()
valid_guesses['valid'] = False
all_words = pd.concat([valid_words, valid_guesses]).reset_index(drop=True)
for i in range(5):
    all_words[f'L-{i}'] = all_words['word'].apply(lambda x: str(x)[i].upper())

letter_cols = [f"L-{_}" for _ in range(5)]
def uniq_cnt(x): return len(list(set(x)))


all_words['unique'] = all_words[letter_cols].apply(uniq_cnt, axis=1)

joblib.dump(all_words, 'all_words.jl')
