import json, sys

with open(sys.argv[1]) as in_f:
    for line in in_f:
        tournament = json.loads(line)
        teams = tournament["Teams"]
        winners = tournament["Winners"]
        maps = tournament["Maps"]
        matchups = (("blue", "red"), ("green", "yellow"), ("white", "black"), \
                    ("purple", "brown"), (winners[0], winners[1]), (winners[2], winners[3]), \
                    (winners[4], winners[5]), (winners[6], "champion"))

        # Clean names from team data.
        for team_color in teams.keys():
            for unit in teams[team_color]["Units"]:
                # Combine "ClassSkills" and "ExtraSkills" into "Abilities"
                unit["Abilities"] = unit["ClassSkills"]
                unit["Abilities"].extend(unit["ExtraSkills"])
                unit.pop("ClassSkills")
                unit.pop("ExtraSkills")
                unit.pop("Name")
                # print(unit)
                # input()
        
        for i, m in enumerate(matchups):
            winner = 0 if winners[i]==m[0] else 1
            match = {"team1": teams[m[0]]["Units"], "team2": teams[m[1]]["Units"], "map": maps[i], "winner": winner}
            print(json.dumps(match))