import json, sys
import requests

get_url = "https://fftbg.com/api/tournaments?limit=500&filter=complete"

while get_url:
    response = requests.get(get_url)
    response_content = response.json()

    for entry in response_content:
        print(json.dumps(entry))
    if 'Link' in response.headers:
        get_url = response.headers['Link'].split('<')[-1].split('>')[0]
    else:
        get_url = None
    print(get_url, file=sys.stderr)