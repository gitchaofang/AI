import regex as re

def get_stats(ids, counts=None): #update counts
    counts = {} if counts is None else counts
    for pair in zip(ids[:-1],ids[1:]):
        counts[pair] = counts.get(pair,0) + 1

def merge(ids,pair,idx):
    newids = []
    i = 0
    while i < len(ids):
        if (ids[i] == pair[0] and i != len(ids) - 1 and ids[i + 1] == pair[1]):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
        

