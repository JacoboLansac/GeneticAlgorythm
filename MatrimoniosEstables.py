import numpy as np
import pandas as pd

N = 3

male_list = ["Male" + str(n) for n in range(N)]
female_list = ["Female" + str(n) for n in range(N)]

male_pref = pd.DataFrame([np.random.choice(np.arange(N), replace=False, size=N) for i in range(N)])
male_pref.index = male_list
male_pref.columns = female_list

female_pref = pd.DataFrame([np.random.choice(np.arange(N), replace=False, size=N) for i in range(N)])
female_pref.index = female_list
female_pref.columns = male_list

couples = pd.Series(index=female_list, data=np.nan, name='Couple')

singles = list(male_pref.index)

while singles:

    print(couples)
    print("Cost function: ???? ")

    candidate = singles.pop()
    candidate_prefs = male_pref.loc[candidate].sort_values().index

    mingled = False
    pref_loc = -1
    while not mingled:

        pref_loc += 1
        female = candidate_prefs[pref_loc]

        if couples.isnull().loc[female]:
            couples.loc[female] = candidate
            mingled = True
        else:
            fem_pref = female_pref.loc[female]
            current_husband = couples.loc[female]
            if fem_pref.loc[candidate] > fem_pref.loc[current_husband]:
                couples.loc[female] = candidate
                singles.append(current_husband)
                mingled = True
            else:
                pass

print(male_pref)
print(female_pref)
print(couples)



