import json, multiprocessing.pool, sys
import requests

ids = []

def make_request(id):
    return requests.get("https://fftbg.com/api/tournament/" + str(id))

with open(sys.argv[1]) as in_f:
    for line in in_f:
        ids.append(json.loads(line)['ID'])

with multiprocessing.pool.ThreadPool(12) as p:
    tournaments = p.map(make_request, ids)

for t in tournaments:
    print(json.dumps(t.json()))
    