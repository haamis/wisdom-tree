import gzip, json, sys
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ordinal encode:
# Sign, Class, ActionSkill, ReactionSkill, SupportSkill, MoveSkill, Mainhand, Offhand, Head, Armor, Accessory, Abilities
# Manually encode / fine as is:
# Gender, Brave, Faith

data = []

with gzip.open(sys.argv[1]) as in_f:
    for line in in_f:
        data.append(json.loads(line))
    # print(json.dumps(data[0], indent=2))

skill_data = []
winner_data = []
for match in data:
    skill_match_data = [] #match_dict = {}
    skill_match_data.extend([match['map']]) # doesn't matter match_dict['map'] = 
    for fighter in match['team1'] + match['team2']:
        skill_match_data.extend([fighter['Gender']]) # doesn't matter
        skill_match_data.extend([fighter['Sign']]) # doesn't matter
        skill_match_data.extend([fighter['Brave']]) # kinda matters
        skill_match_data.extend([fighter['Faith']]) # kinda matters
        skill_match_data.extend([fighter['Class']]) # really matters
        skill_match_data.extend([fighter['Mainhand'] if fighter['Mainhand'] else None]) # really matters
        skill_match_data.extend([fighter['Offhand'] if fighter['Offhand'] else None]) # matters
        skill_match_data.extend([fighter['Head'] if fighter['Head'] else None]) # kinda matters
        skill_match_data.extend([fighter['Armor'] if fighter['Armor'] else None]) # kinda matters 
        skill_match_data.extend([fighter['Accessory'] if fighter['Accessory'] else None]) # kinda matters

# Class predictive (60%), mainhand very close, offhand not as close, brave and faith also kinda predictive

        skills = (fighter['ActionSkill'], fighter['ReactionSkill'], \
                    fighter['SupportSkill'], fighter['MoveSkill'])
        skill_list = [None]*4
        for i, skill in enumerate(skills):
            if skill:
                skill_list[i] = skill
        skill_match_data.extend(skill_list)
        abilities = [None]*30
        for i, ability in enumerate(fighter['Abilities']):
            if ability:
                abilities[i] = ability
        skill_match_data.extend(abilities)

    skill_data.append(skill_match_data)
    winner_data.append(match['winner'])

feature_labels = [
    "map", 

    "t1p1_gender", "t1p1_sign", "t1p1_brave", "t1p1_faith", "t1p1_class", 
    "t1p1_mainhand", "t1p1_offhand", "t1p1_head", "t1p1_armor", "t1p1_accessory", 
    "t1p1_actionskill", "t1p1_reactionskill", "t1p1_supportskill", "t1p1_moveskill",
    "t1p1_ability1", "t1p1_ability2", "t1p1_ability3", "t1p1_ability4", "t1p1_ability5", 
    "t1p1_ability6", "t1p1_ability7", "t1p1_ability8", "t1p1_ability9", "t1p1_ability10", 
    "t1p1_ability11", "t1p1_ability12", "t1p1_ability13", "t1p1_ability14", "t1p1_ability15", 
    "t1p1_ability16", "t1p1_ability17", "t1p1_ability18", "t1p1_ability19", "t1p1_ability20", 
    "t1p1_ability21", "t1p1_ability22", "t1p1_ability23", "t1p1_ability24", "t1p1_ability25", 
    "t1p1_ability26", "t1p1_ability27", "t1p1_ability28", "t1p1_ability29", "t1p1_ability30",

    "t1p2_gender", "t1p2_sign", "t1p2_brave", "t1p2_faith", "t1p2_class", 
    "t1p2_mainhand", "t1p2_offhand", "t1p2_head", "t1p2_armor", "t1p2_accessory", 
    "t1p2_actionskill", "t1p2_reactionskill", "t1p2_supportskill", "t1p2_moveskill",
    "t1p2_ability1", "t1p2_ability2", "t1p2_ability3", "t1p2_ability4", "t1p2_ability5", 
    "t1p2_ability6", "t1p2_ability7", "t1p2_ability8", "t1p2_ability9", "t1p2_ability10", 
    "t1p2_ability11", "t1p2_ability12", "t1p2_ability13", "t1p2_ability14", "t1p2_ability15", 
    "t1p2_ability16", "t1p2_ability17", "t1p2_ability18", "t1p2_ability19", "t1p2_ability20", 
    "t1p2_ability21", "t1p2_ability22", "t1p2_ability23", "t1p2_ability24", "t1p2_ability25", 
    "t1p2_ability26", "t1p2_ability27", "t1p2_ability28", "t1p2_ability29", "t1p2_ability30",
    
    "t1p3_gender", "t1p3_sign", "t1p3_brave", "t1p3_faith", "t1p3_class", 
    "t1p3_mainhand", "t1p3_offhand", "t1p3_head", "t1p3_armor", "t1p3_accessory", 
    "t1p3_actionskill", "t1p3_reactionskill", "t1p3_supportskill", "t1p3_moveskill",
    "t1p3_ability1", "t1p3_ability2", "t1p3_ability3", "t1p3_ability4", "t1p3_ability5", 
    "t1p3_ability6", "t1p3_ability7", "t1p3_ability8", "t1p3_ability9", "t1p3_ability10", 
    "t1p3_ability11", "t1p3_ability12", "t1p3_ability13", "t1p3_ability14", "t1p3_ability15", 
    "t1p3_ability16", "t1p3_ability17", "t1p3_ability18", "t1p3_ability19", "t1p3_ability20", 
    "t1p3_ability21", "t1p3_ability22", "t1p3_ability23", "t1p3_ability24", "t1p3_ability25", 
    "t1p3_ability26", "t1p3_ability27", "t1p3_ability28", "t1p3_ability29", "t1p3_ability30",
    
    "t1p4_gender", "t1p4_sign", "t1p4_brave", "t1p4_faith", "t1p4_class", 
    "t1p4_mainhand", "t1p4_offhand", "t1p4_head", "t1p4_armor", "t1p4_accessory", 
    "t1p4_actionskill", "t1p4_reactionskill", "t1p4_supportskill", "t1p4_moveskill",
    "t1p4_ability1", "t1p4_ability2", "t1p4_ability3", "t1p4_ability4", "t1p4_ability5", 
    "t1p4_ability6", "t1p4_ability7", "t1p4_ability8", "t1p4_ability9", "t1p4_ability10", 
    "t1p4_ability11", "t1p4_ability12", "t1p4_ability13", "t1p4_ability14", "t1p4_ability15", 
    "t1p4_ability16", "t1p4_ability17", "t1p4_ability18", "t1p4_ability19", "t1p4_ability20", 
    "t1p4_ability21", "t1p4_ability22", "t1p4_ability23", "t1p4_ability24", "t1p4_ability25", 
    "t1p4_ability26", "t1p4_ability27", "t1p4_ability28", "t1p4_ability29", "t1p4_ability30",

    "t2p1_gender", "t2p1_sign", "t2p1_brave", "t2p1_faith", "t2p1_class", 
    "t2p1_mainhand", "t2p1_offhand", "t2p1_head", "t2p1_armor", "t2p1_accessory", 
    "t2p1_actionskill", "t2p1_reactionskill", "t2p1_supportskill", "t2p1_moveskill",
    "t2p1_ability1", "t2p1_ability2", "t2p1_ability3", "t2p1_ability4", "t2p1_ability5", 
    "t2p1_ability6", "t2p1_ability7", "t2p1_ability8", "t2p1_ability9", "t2p1_ability10", 
    "t2p1_ability11", "t2p1_ability12", "t2p1_ability13", "t2p1_ability14", "t2p1_ability15", 
    "t2p1_ability16", "t2p1_ability17", "t2p1_ability18", "t2p1_ability19", "t2p1_ability20", 
    "t2p1_ability21", "t2p1_ability22", "t2p1_ability23", "t2p1_ability24", "t2p1_ability25", 
    "t2p1_ability26", "t2p1_ability27", "t2p1_ability28", "t2p1_ability29", "t2p1_ability30",

    "t2p2_gender", "t2p2_sign", "t2p2_brave", "t2p2_faith", "t2p2_class", 
    "t2p2_mainhand", "t2p2_offhand", "t2p2_head", "t2p2_armor", "t2p2_accessory", 
    "t2p2_actionskill", "t2p2_reactionskill", "t2p2_supportskill", "t2p2_moveskill",
    "t2p2_ability1", "t2p2_ability2", "t2p2_ability3", "t2p2_ability4", "t2p2_ability5", 
    "t2p2_ability6", "t2p2_ability7", "t2p2_ability8", "t2p2_ability9", "t2p2_ability10", 
    "t2p2_ability11", "t2p2_ability12", "t2p2_ability13", "t2p2_ability14", "t2p2_ability15", 
    "t2p2_ability16", "t2p2_ability17", "t2p2_ability18", "t2p2_ability19", "t2p2_ability20", 
    "t2p2_ability21", "t2p2_ability22", "t2p2_ability23", "t2p2_ability24", "t2p2_ability25", 
    "t2p2_ability26", "t2p2_ability27", "t2p2_ability28", "t2p2_ability29", "t2p2_ability30",

    "t2p3_gender", "t2p3_sign", "t2p3_brave", "t2p3_faith", "t2p3_class", 
    "t2p3_mainhand", "t2p3_offhand", "t2p3_head", "t2p3_armor", "t2p3_accessory", 
    "t2p3_actionskill", "t2p3_reactionskill", "t2p3_supportskill", "t2p3_moveskill",
    "t2p3_ability1", "t2p3_ability2", "t2p3_ability3", "t2p3_ability4", "t2p3_ability5", 
    "t2p3_ability6", "t2p3_ability7", "t2p3_ability8", "t2p3_ability9", "t2p3_ability10", 
    "t2p3_ability11", "t2p3_ability12", "t2p3_ability13", "t2p3_ability14", "t2p3_ability15", 
    "t2p3_ability16", "t2p3_ability17", "t2p3_ability18", "t2p3_ability19", "t2p3_ability20", 
    "t2p3_ability21", "t2p3_ability22", "t2p3_ability23", "t2p3_ability24", "t2p3_ability25", 
    "t2p3_ability26", "t2p3_ability27", "t2p3_ability28", "t2p3_ability29", "t2p3_ability30",

    "t2p4_gender", "t2p4_sign", "t2p4_brave", "t2p4_faith", "t2p4_class", 
    "t2p4_mainhand", "t2p4_offhand", "t2p4_head", "t2p4_armor", "t2p4_accessory", 
    "t2p4_actionskill", "t2p4_reactionskill", "t2p4_supportskill", "t2p4_moveskill",
    "t2p4_ability1", "t2p4_ability2", "t2p4_ability3", "t2p4_ability4", "t2p4_ability5", 
    "t2p4_ability6", "t2p4_ability7", "t2p4_ability8", "t2p4_ability9", "t2p4_ability10", 
    "t2p4_ability11", "t2p4_ability12", "t2p4_ability13", "t2p4_ability14", "t2p4_ability15", 
    "t2p4_ability16", "t2p4_ability17", "t2p4_ability18", "t2p4_ability19", "t2p4_ability20", 
    "t2p4_ability21", "t2p4_ability22", "t2p4_ability23", "t2p4_ability24", "t2p4_ability25", 
    "t2p4_ability26", "t2p4_ability27", "t2p4_ability28", "t2p4_ability29", "t2p4_ability30",
]

print(skill_data[0])
skill_df = pd.DataFrame(data=skill_data)#, dtype='category')
print([x for x in skill_df.iloc[0]])

# to_ord_enc = 

# one_hotter = OneHotEncoder()
# skill_df = one_hotter.fit_transform(skill_df)

ordinal_enc = OrdinalEncoder()
skill_df = ordinal_enc.fit_transform(skill_df)

train_df, valid_df, train_winners, valid_winners = train_test_split(skill_df, winner_data, test_size=0.1)
print(train_df.shape)

rf = RandomForestClassifier(2000, criterion='entropy', n_jobs=-1, verbose=2)
rf.fit(train_df, train_winners)
print("finished fitting, scoring..")
print(rf.score(valid_df, valid_winners))
