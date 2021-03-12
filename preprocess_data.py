import json, sys
from random import shuffle
import lightgbm as lgb
import matplotlib.pyplot as plt
import scipy
from xopen import xopen
import numpy as np
import pandas as pd
from pandas.core.base import DataError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Ordinal encode:
# Sign, Class, ActionSkill, ReactionSkill, SupportSkill, MoveSkill, Mainhand, Offhand, Head, Armor, Accessory, Abilities
# Manually encode / fine as is:
# Gender, Brave, Faith

WITH_ABILITIES = True

def yield_data():
    with xopen(sys.argv[1]) as in_f:
        for line in in_f:
            yield json.loads(line)
    # print(json.dumps(data[0], indent=2))

# Class predictive (60% acc w/ class only), mainhand very close, offhand not as close, brave and faith also kinda predictive
def parse_data(data):
    skill_data = []
    ability_data = []
    ability_lens = []
    winner_data = []
    for match in data:
        skill_match_data = []
        # ability_match_data = []
        skill_match_data.extend([int(match["map"].split(")")[0])]) # maybe matters a lot?
        for fighter in match['team1'] + match['team2']:
            
            if fighter['Gender'] == 'Male':
                skill_match_data.extend([0])
            elif fighter['Gender'] == 'Female':
                skill_match_data.extend([1])
            elif fighter['Gender'] == 'Monster':
                skill_match_data.extend([2])
            else:
                raise NameError('Unknown gender', fighter['Gender'])
            
            skill_match_data.extend([fighter['Sign']]) # doesn't matter
            # change manually to numerical instead of categorical?
            skill_match_data.extend([fighter['Brave']]) # kinda matters
            skill_match_data.extend([fighter['Faith']]) # kinda matters
            skill_match_data.extend([fighter['Class']]) # really matters
            
            skill_match_data.extend([fighter['Mainhand'] if fighter['Mainhand'] else None]) # really matters
            skill_match_data.extend([fighter['Offhand'] if fighter['Offhand'] else None]) # matters
            skill_match_data.extend([fighter['Head'] if fighter['Head'] else None]) # kinda matters
            skill_match_data.extend([fighter['Armor'] if fighter['Armor'] else None]) # kinda matters 
            skill_match_data.extend([fighter['Accessory'] if fighter['Accessory'] else None]) # kinda matters
            
            skill_match_data.extend([fighter['ActionSkill'] if fighter['ActionSkill'] else None])
            skill_match_data.extend([fighter['ReactionSkill'] if fighter['ReactionSkill'] else None])
            skill_match_data.extend([fighter['SupportSkill'] if fighter['SupportSkill'] else None])
            skill_match_data.extend([fighter['MoveSkill'] if fighter['MoveSkill'] else None])

            # OLD CODE!
            # skills = (fighter['ActionSkill'], fighter['ReactionSkill'], \
            #             fighter['SupportSkill'], fighter['MoveSkill'])
            # skill_list = [None]*4
            # for i, skill in enumerate(skills):
            #     if skill:
            #         skill_list[i] = skill
            # skill_match_data.extend(skill_list)

            # ability_list = [a for a in fighter['Abilities']]

            # ability_list = [None]*30
            # for i, ability in enumerate(fighter['Abilities']):
            #     if ability:
            #         ability_list[i] = ability

            if WITH_ABILITIES:
                i = 0
                for i, ability in enumerate(fighter['Abilities']):
                    ability_data.append(ability)
                
                ability_lens.append(i)#len(ability_list))
            # ability_data.append(ability_list)
            
            # DEBUG
            # print(skill_match_data)
            # print(ability_match_data)
            # input()

        # # Only without skill_list!
        # if len(skill_match_data) != 113:
        #     raise DataError('len(skill_match_data) not 113, instead', len(skill_match_data))
        skill_data.append(skill_match_data)
        winner_data.append(match['winner'])
    return skill_data, winner_data, ability_data, ability_lens

skill_data, winner_data, ability_data, ability_lens = parse_data(yield_data())

skill_data = np.array(skill_data)
print(skill_data.shape)

col_len = 14

# This memory hog(?) needs to be torn down.
# Also should do features per column so they can be deleted after transformation instead of all at once.
# Either way, peak memory usage needs to go down.
col_transformer = ColumnTransformer([
    ('Passthrough', 'passthrough', [0] + [ (n*col_len)+x for n in range(8) for x in (1,3,4) ] ),
    ('Sign', OrdinalEncoder(), [ (n*col_len)+2 for n in range(8) ] ),
    ('Class', OrdinalEncoder(), [ (n*col_len)+5 for n in range(8) ] ),
    ('Mainhand', OrdinalEncoder(), [ (n*col_len)+6 for n in range(8) ] ),
    ('Offhand', OrdinalEncoder(), [ (n*col_len)+7 for n in range(8) ] ) ,
    ('Head', OrdinalEncoder(), [ (n*col_len)+8 for n in range(8) ] ),
    ('Armor', OrdinalEncoder(), [ (n*col_len)+9 for n in range(8) ] ),
    ('Accessory', OrdinalEncoder(), [ (n*col_len)+10 for n in range(8) ] ),
    ('ActionSkill', OrdinalEncoder(), [ (n*col_len)+11 for n in range(8) ] ),
    ('ReactionSkill', OrdinalEncoder(), [ (n*col_len)+12 for n in range(8) ] ),
    ('SupportSkill', OrdinalEncoder(), [ (n*col_len)+13 for n in range(8) ] ),
    ('MoveSkill', OrdinalEncoder(), [ (n*col_len)+14 for n in range(8) ] ),
])

print('start col_transform')
print(skill_data[0])
skill_data = col_transformer.fit_transform(skill_data).astype('float')
print(skill_data[0])
print(skill_data.shape)

# Try without one hotting again, would probably reduce memory usage by a lot.
if WITH_ABILITIES:
    print('start building ability array')
    ability_data = pd.DataFrame(ability_data)
    print(ability_data[0])

    one_hotter = OneHotEncoder()
    print('start one_hotter')
    ability_data = one_hotter.fit_transform(ability_data)

    print(type(ability_data))

    print(ability_data.shape)
    print(ability_data[0:13])
    #new_ability_data = scipy.sparse.lil_matrix((1, len(one_hotter.categories_)))
    # new_ability_data = np.zeros_like(ability_data[0])
    ab_index = 0
    #temp = np.zeros_like(ability_data[0])
    ability_data.astype('uint8')
    one_hots = []
    print(np.sum(ability_data[ab_index : ab_index+0], axis=0).shape)
    for i, entry in enumerate(ability_lens):
        # for j in range(entry):
        #     temp = temp | ability_data[ab_index]
        #     ab_index += 1
        # new_ability_data = np.vstack((new_ability_data, np.sum(ability_data[ab_index : ab_index+entry])))
        # vstacking seems slow (memcopies probably), collect a list, stack all at once later.
        # TODO: collect a list of sparse matrices and then vstack them?
        one_hots.append(np.sum(ability_data[ab_index : ab_index+entry], axis=0))
        ab_index += entry
        # np.vstack((new_ability_data, temp)) # scipy.sparse
        # temp = np.zeros_like(ability_data[0])
    new_ability_data = np.vstack(one_hots)
    del one_hots
    new_ability_data = np.clip(new_ability_data, 0.0, 1.0)
    ability_data = scipy.sparse.csr_matrix(new_ability_data.reshape((skill_data.shape[0], -1)))
    print(ability_data[0])
    print(ability_data.shape)
    print(ability_lens[0:8])
    #print(ability_data.tocsr()[0])

    del new_ability_data
    del ability_lens

    input_data = scipy.sparse.hstack((skill_data, ability_data)).astype('float').tocsr()
    del ability_data

else:
    input_data = scipy.sparse.csr_matrix(skill_data)

# Saving that memory.
# del new_ability_data
# del one_hots
del skill_data
# del ability_data
# del ability_lens
# del data

# UNUSED ATM!
# skill_df = pd.DataFrame(data=skill_data, dtype='category') # SparseDtype()
# skill_df = pd.DataFrame.astype(skill_df, dtype=SparseDtype('category'))
# skill_df = pd.concat((skill_df, ability_one_hot), axis=1, ignore_index=True)
# print([x for x in skill_df.iloc[0]])
# print(skill_df.iloc[0][18])

# UNUSED! Don't treat any brave or faith values as categorical variables. This line is maybe too cute.
# categorical_features = [x for x in range(len(skill_data[0])) if x not in set( [ (n*15)+3 for n in range(8) ] + [ (n*15)+4 for n in range(8) ] )]
# categorical_features = [x for x in range(len(skill_data[0]))]
# for num in [(n*15)+3 for n in range(8)] + [(n*15)+4 for n in range(8)]:
#     categorical_features.remove(num)


feature_labels = [
    "map", 

    "t1p1_gender", "t1p1_brave", "t1p1_faith", "t1p1_sign", "t1p1_class", 
    "t1p1_mainhand", "t1p1_offhand", "t1p1_head", "t1p1_armor", "t1p1_accessory", 
    "t1p1_actionskill", "t1p1_reactionskill", "t1p1_supportskill", "t1p1_moveskill",
    
    "t1p2_gender", "t1p2_brave", "t1p2_faith", "t1p2_sign", "t1p2_class", 
    "t1p2_mainhand", "t1p2_offhand", "t1p2_head", "t1p2_armor", "t1p2_accessory", 
    "t1p2_actionskill", "t1p2_reactionskill", "t1p2_supportskill", "t1p2_moveskill",
    
    "t1p3_gender", "t1p3_brave", "t1p3_faith", "t1p3_sign", "t1p3_class", 
    "t1p3_mainhand", "t1p3_offhand", "t1p3_head", "t1p3_armor", "t1p3_accessory", 
    "t1p3_actionskill", "t1p3_reactionskill", "t1p3_supportskill", "t1p3_moveskill",
    
    "t1p4_gender", "t1p4_brave", "t1p4_faith", "t1p4_sign", "t1p4_class", 
    "t1p4_mainhand", "t1p4_offhand", "t1p4_head", "t1p4_armor", "t1p4_accessory", 
    "t1p4_actionskill", "t1p4_reactionskill", "t1p4_supportskill", "t1p4_moveskill",
    
    "t2p1_gender", "t2p1_brave", "t2p1_faith", "t2p1_sign", "t2p1_class", 
    "t2p1_mainhand", "t2p1_offhand", "t2p1_head", "t2p1_armor", "t2p1_accessory", 
    "t2p1_actionskill", "t2p1_reactionskill", "t2p1_supportskill", "t2p1_moveskill",
   
    "t2p2_gender", "t2p2_brave", "t2p2_faith", "t2p2_sign", "t2p2_class", 
    "t2p2_mainhand", "t2p2_offhand", "t2p2_head", "t2p2_armor", "t2p2_accessory", 
    "t2p2_actionskill", "t2p2_reactionskill", "t2p2_supportskill", "t2p2_moveskill",
   
    "t2p3_gender", "t2p3_brave", "t2p3_faith", "t2p3_sign", "t2p3_class", 
    "t2p3_mainhand", "t2p3_offhand", "t2p3_head", "t2p3_armor", "t2p3_accessory", 
    "t2p3_actionskill", "t2p3_reactionskill", "t2p3_supportskill", "t2p3_moveskill",
   
    "t2p4_gender", "t2p4_brave", "t2p4_faith", "t2p4_sign", "t2p4_class", 
    "t2p4_mainhand", "t2p4_offhand", "t2p4_head", "t2p4_armor", "t2p4_accessory", 
    "t2p4_actionskill", "t2p4_reactionskill", "t2p4_supportskill", "t2p4_moveskill",
]

feature_names = ['map']

for team in ('t1','t2'):
    for player in ('p1','p2','p3','p4'):
        for feature in ('gender', 'brave', 'faith'):
            feature_names.append(team+player+feature)

for feature in ('sign', 'class', 'mainhand', 'offhand', 'head', 'armor', 'accessory',
                'actionskill', 'reactionskill', 'supportskill', 'moveskill'):
    for team in ('t1','t2'):
        for player in ('p1','p2','p3','p4'):
            feature_names.append(team+player+feature)

if WITH_ABILITIES:
    ability_names = []
    for team in ('t1','t2'):
        for player in ('p1','p2','p3','p4'):
            for ability in list(one_hotter.categories_[0]):
                ability_names.append(team+player+ability)
    
    feature_labels = feature_names + ability_names
    del ability_names

else:
    feature_labels = feature_names

del feature_names

categorical_features = [0] + [x for x in range(25,input_data.shape[1]+1)]

# Normalize and standardize data, hopefully get some score improvements.
# scaler = MaxAbsScaler()
# input_data = scaler.fit_transform(input_data)
# scaler = StandardScaler(with_mean=False)
# input_data = scaler.fit_transform(input_data)

print(input_data)

num_categories = np.amax(input_data, axis=0).toarray()[0]#.astype('int')
num_categories += 1
if WITH_ABILITIES:
    num_categories[113:] = 2
print(num_categories, num_categories.shape)
num_categories = [int(c) for c in num_categories]
print(num_categories, type(num_categories), type(num_categories[0]))

NUM_SAMPLES_TEST = 16958

train_data = lgb.Dataset(input_data[NUM_SAMPLES_TEST:], label=winner_data[NUM_SAMPLES_TEST:], feature_name=feature_labels, params={'max_bin_by_feature': num_categories}, categorical_feature=categorical_features,)

validation_data = train_data.create_valid(input_data[:NUM_SAMPLES_TEST], label=winner_data[:NUM_SAMPLES_TEST])

train_data.save_binary("train.bin")
validation_data.save_binary("val.bin")

with open("num_categories.jsonl", "w") as out_f:
    json.dump(num_categories, out_f)