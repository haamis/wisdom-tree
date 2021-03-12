import json, sys
from random import shuffle
from itertools import permutations
from multiprocessing import Pool, cpu_count
from xopen import xopen

MIRROR = False
# Number of augmented matches to return
NUM_AUG_PER_MATCH = 1
CPU_COUNT = cpu_count()

def yield_data():
    with xopen(sys.argv[1]) as in_f:
        for line in in_f:
            yield json.loads(line)

def inner_perms(match):
    ret = []
    team1_perms = list(permutations(match['team1']))
    team2_perms = list(permutations(match['team2']))

    for t1 in team1_perms:
        for t2 in team2_perms:
            ret.append(json.dumps({'team1': t1, 'team2': t2, 'map': match['map'], 'winner': match['winner']}))
            # Flip winner bit and swap t1 and t2 places.
            if MIRROR:
                ret.append(json.dumps({'team1': t2, 'team2': t1, 'map': match['map'], 'winner': int(not match['winner'])}))
    
    # Ignore case of first permutations of both teams as this is the original match.
    ret = ret[1:]
    shuffle(ret)
    return ret[:NUM_AUG_PER_MATCH]

def yield_aug(data):
    with Pool(CPU_COUNT) as p:
        matches = []
        for match in data:
            matches.append(match)
            if len(matches) == CPU_COUNT:
                yield_items = p.map(inner_perms, matches)
                # Yield one match a time.
                # print("yield_items:", len(yield_items), file=sys.stderr)
                for item in yield_items:
                    yield item
                matches = []

for match_perms in yield_aug(yield_data()):
    # print("match_perms:", len(match_perms), file=sys.stderr)
    for match in match_perms:
        print(match)