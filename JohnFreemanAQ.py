import numpy as np
import pandas as pd
from numbers import Number
import sys
from multiprocessing import Pool
from functools import partial
import os.path
import time
import itertools


temp_str = ""
temp_str2 = ""

maxstar = 5

file_name = 'Notes1.txt'

file_name = input("Please enter an input file\n > ")

while not os.path.isfile(file_name):
    file_name = input("Sorry, that file can't be read, try again\n > ")

maxstar = input("Plase enter the maxstar parameter\n > ")

while True:
    if maxstar.isdigit():
            if int(maxstar) != 0:
                break
    maxstar = input("Sorry, '" + maxstar + "' doesn't seem to be a valid positive nonzero int, try again\n > ")

maxstar = int(maxstar)

with open(file_name, 'r') as my_file:
    data = my_file.read()

i = 0
state = 0
cur_iter = 0

print("Parsing input file")
while i < len(data) - 1:
    if data[i] == '!':
        comment = i
        while data[comment] != '\n':
            if comment >= len(data):
                break
            comment += 1

        data = data[:i] + data[comment:]

    if state == 0 and data[i] == '>':
        length = len([x for x in data[:i].split() if not x in '<>'])
        state = 1
        end_one = i + 1
    elif state == 1 and data[i] == ']':
        attrs = [x for x in data[end_one:i].split() if not x in '[]']
        target = attrs[-1]
        attrs = attrs
        state = 2
        vals_arr = [[] for x in range(length)]
    elif state == 2:
        temp_str = ""
        if cur_iter == length:
            cur_iter = 0

        while i < len(data) and data[i].isspace():
            i += 1

        if i >= len(data):
            break

        while i < len(data) and not data[i].isspace():
            temp_str += data[i]
            i += 1

        if i >= len(data):
            break

        if cur_iter == length - 1:
            vals_arr[cur_iter].append(temp_str)

        else:
            try:
                temp_str = float(temp_str)
            except ValueError:
                temp_str = temp_str

            vals_arr[cur_iter].append(temp_str)

        cur_iter += 1

    i += 1

vals_arr = [list(x) for x in zip(*vals_arr)]

df_test = pd.DataFrame(vals_arr, columns=attrs)

def cutpointify(numeric_col):
    if not isinstance(numeric_col[0], Number):
        numeric_df = numeric_col.to_frame()
        return numeric_df

    uniques = numeric_col.unique()
    uniques = sorted(uniques)
    cutpoints = [(uniques[i] + uniques[i+1])/2 for i in range(len(uniques) - 1)]
    ranges = []
    for i in range(len(cutpoints)):
        ranges.append(((uniques[0], cutpoints[i]), (cutpoints[i], uniques[-1])))

    new_names = []

    for cutpoint in cutpoints:
        new_names.append(numeric_col.name + str(cutpoint))

    new_list = [[] for i in range(len(new_names))]

    for idx, val in numeric_col.iteritems():
        for i in range(len(ranges)):
            if (val < ranges[i][0][1]):
                new_list[i].append(str(ranges[i][0][0]) + ".." + str(ranges[i][0][1]))
            else:
                new_list[i].append(str(ranges[i][1][0]) + ".." + str(ranges[i][1][1]))

    new_list = [list(x) for x in zip(*new_list)]

    new_df = pd.DataFrame(new_list, columns=new_names)
    return new_df


print("Creating cutpointed columns")
cutpointed = []
for i in range(len(df_test.columns) - 1):
    cutpointed.append(cutpointify(df_test[df_test.columns[i]]))

cutpointed.append(df_test[df_test.columns[-1]].to_frame())
df_test = pd.concat(cutpointed, axis=1)

attrs = list(df_test.columns[:-1])

df_target_dropped = df_test.drop(labels=target, axis=1)
df_duplicates = df_target_dropped.loc[df_target_dropped.duplicated(keep=False)]
df_duplicates_with_target = df_test.loc[list(df_duplicates.index)]

print("Checking for consistency")
for idx, row in df_duplicates.iterrows():
    for idx2, row2 in df_duplicates.iterrows():
        if row.equals(row2):
            if not df_duplicates_with_target.loc[idx].equals(df_duplicates_with_target.loc[idx2]):
                print("The dataset is inconsistent!")
                out_file = open("my-data.without.negation.rul", "w")
                out_file.write("! The input data set is inconsistent")
                out_file.close()
                out_file = open("my-data.with.negation.rul", "w")
                out_file.write("! The input data set is inconsistent")
                out_file.close()
                sys.exit()

def elementary_star(e_seed, e_not_seed):
    rules = []
    for idx, val in e_seed.iteritems():
        if e_seed[idx] != e_not_seed[idx]:
            rules.append([(idx, e_not_seed[idx])])
    return rules

def covers(partial_star, seed):
    match_flag = True
    covered = []
    for idx, row in df_test.iterrows():
        for rules in partial_star:
            match_flag = True
            for rule in rules:
                if row[rule[0]] == rule[1]:
                    match_flag = False
                    break
            if match_flag:
                covered.append(idx)
                if (row[target] != seed):
                    return [idx]
                break

    return covered

def remove_duplicates(ls):
    seen = set()
    return [x for x in ls if not (tuple(x) in seen or seen.add(tuple(x)))]

def remove_supersets(star):
    if all(len(x) == len(star[0]) for x in star):
        return remove_duplicates(star)
    s = 0
    star = remove_duplicates(star)
    sort = sorted(star, key=len)

    while s < len(star):
        if all(len(x) == len(sort[s]) for x in sort[s:]):
            return star
        supersets = [x for x in star if set(sort[s]) < set(x)]
        star[:] = [x for x in star if x not in supersets]
        sort[:] = [x for x in sort if x not in supersets]
        s += 1

    return star

def add_to_rules(rule, rules):
    rules.append(rule)
    return rules

def merge_star(star1s, star2s):
    if not star1s or all(x == [] for x in star1s):
        return star2s
    if not star2s or all(x == [] for x in star2s):
        return star1s
    ret_star = []
    star1s[:] = remove_supersets(star1s)
    star2s[:] = remove_supersets(star2s)

    longest = -1

    star1s_iter = 0

    while (star1s_iter < len(star1s)):
        star2s_iter = 0
        while (star2s_iter < len(star2s)):
            if set(star1s[star1s_iter]) == set(star2s[star2s_iter]):
                add_to_rules(star1s[star1s_iter], ret_star)
                star2s.remove(star1s[star1s_iter])
                break
            elif set(star1s[star1s_iter]).issubset(set(star2s[star2s_iter])):
                add_to_rules(star2s[star2s_iter], ret_star)
            elif set(star2s[star2s_iter]).issubset(set(star1s[star1s_iter])):
                add_to_rules(star1s[star1s_iter], ret_star)
            else:
                new_rule = star1s[star1s_iter] + star2s[star2s_iter]
                add_to_rules(new_rule, ret_star)

            star2s_iter += 1
        star1s_iter += 1

    ret_star = remove_supersets(ret_star)
    return ret_star

def trim_rule(partial_star, seed):
    covered = [len(covers([star], seed)) for star in partial_star]
    while len(partial_star) > maxstar:
        worst = min(covered)
        if all(x == worst for x in covered):
            return partial_star[:maxstar]
        partial_star[:] = [partial_star[x] for x in range(len(partial_star)) if covered[x] != worst]
        covered = [covered[x] for x in range(len(covered)) if covered[x] != worst]

    return partial_star

def linear_dropping(partial_star, seed):
    if not partial_star or not partial_star[0]:
        return partial_star

    for star in range(len(partial_star)):
        covered = covers([partial_star[star]], seed)
        s = 0
        while s < len(partial_star[star]):
            test_star = [x for x in partial_star[star] if x != partial_star[star][s]]
            test_covers = covers([test_star], seed)
            if set(test_covers) >= set(covered):
                if all(df_test.loc[x][target] == seed for x in set(test_covers)):
                    partial_star[star][:] = test_star
                    s -= 1
            s += 1

    return partial_star

def negate_star(df, star):
    for i in range(len(star)):
        for j in range(len(star[i])):
            vals = list(df_test[star[i][j][0]].unique())
            vals.remove(star[i][j][1])
            negated = ' v '.join(vals)
            star[i][j] = list(star[i][j])
            star[i][j][1] = negated
            star[i][j] = tuple(star[i][j])

    return star

def format_output_rule_without(seed, rules):
    temp_str = ""
    for rule in rules:
        temp_str = temp_str + "(" + rule[0] + ", " + rule[1] + ") & "
    temp_str = temp_str[:-3]
    temp_str = temp_str + " -> (" + target + ", " + seed + ")\n"
    return temp_str

def format_output_rule(seed, rules):
    temp_str = ""
    for rule in rules:
        temp_str = temp_str + "(" + rule[0] + ", not " + rule[1] + ") & "
    temp_str = temp_str[:-3]
    temp_str = temp_str + " -> (" + target + ", " + seed + ")\n"
    return temp_str

all_stars = []

examples_covered = 0.0

initial_len = len(list(df_test.index))

for seed in df_test[target].unique():

    df_seed = df_test.loc[df_test[target] == seed]
    df_seed = df_seed.drop(labels=target, axis=1)

    df_not_seed = df_test.loc[df_test[target] != seed]
    df_not_seed = df_not_seed.drop(labels=target, axis=1)

    seeds = list(df_seed.index)
    not_seeds = list(df_not_seed.index)
    stars = []

    cur_star = elementary_star(df_seed.loc[seeds[0]], df_not_seed.loc[not_seeds[0]])

    sys.stdout.flush()

    percent = 100*examples_covered/initial_len
    percent = '%.3f'%(percent)

    sys.stdout.write('\rComputing Rules: ' + percent + '%')

    while len(seeds) != 0:
        covered = covers(cur_star, seed)
        if all(df_test.loc[x][target] == seed for x in covered):
            dropped = linear_dropping(cur_star, seed)
            covered = covers(dropped, seed)

            before = len(seeds)
            seeds[:] = [x for x in seeds if x not in covered]
            after = len(seeds)

            examples_covered += before
            examples_covered -= after
            stars.append(dropped)

            sys.stdout.flush()
            percent = 100*examples_covered/initial_len
            percent = '%.3f'%(percent)
            sys.stdout.write('\rComputing Rules: ' + percent + '%')

            if len(seeds) != 0:
                cur_star = elementary_star(df_seed.loc[seeds[0]], df_not_seed.loc[not_seeds[0]])
        else:
            covered_negatives = [x for x in covered if x in not_seeds]
            new_star = elementary_star(df_seed.loc[seeds[0]], df_not_seed.loc[covered_negatives[0]])
            cur_star = merge_star(cur_star, new_star)
            cur_star = trim_rule(cur_star, seed)

    all_stars.append((seed, stars))

out_file = open("my-data.with.negation.rul", "w")

for stars in all_stars:
    for star in stars[1]:
        for s in star:
            out_file.write(format_output_rule(stars[0], s))

out_file.close()

negated_all_stars = all_stars[:]

for stars in negated_all_stars:
    for i in range(len(stars[1])):
        stars[1][i] = negate_star(df_test, stars[1][i])

out_file = open("my-data.without.negation.rul", "w")

for stars in negated_all_stars:
    for star in stars[1]:
        for s in star:
            out_file.write(format_output_rule_without(stars[0], s))

out_file.close()

print("\nRules successfully generated")
