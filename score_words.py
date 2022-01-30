"""
Score the set of candidate words extensively
"""
import joblib
import pandas as pd
import wordlexp as we
import multiprocessing

if __name__ == '__main__':
    #candidates = ['irate', 'later', 'inter', 'liner', 'alert', 'tails', 'trail', 'arose', 'torus',
    #              'bonus', 'saner', 'boney', 'focus']
    #candidates = candidates[:4]
    candidates = list(we.all_words[(we.all_words["valid"]) & (we.all_words["unique"] == 5)]["word"])
    print("checking", len(candidates))
    with multiprocessing.Pool(4) as pool:
        chunks = pool.map(we.evaluate_first_word, candidates)#we.all_words[we.all_words["valid"]]["word"])
        pool.close()
        pool.join()
    joblib.dump(pd.DataFrame(chunks), 'candidate_scores.jl')
