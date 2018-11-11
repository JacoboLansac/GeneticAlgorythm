import sys
import math
import numpy as np
import pandas as pd

# ===================================================================================
#####################################################################################
# Foo example with few trafic lights
#####################################################################################
# ===================================================================================
times = [10, 12]  # secs
distances = [25, 90]  # meters

total_dist = 100

v = 1 # m/s
t_end = total_dist / v
t = np.arange(0, t_end, 1)

greens = []
for tf, df in zip(times, distances):
    s = pd.DataFrame(index=t, data=(t % (2*tf)))
    green = (s < tf).astype(int)
    greens.append(green)
    # s.green.plot()
df = pd.concat(greens, axis=1)
df.plot()
sum = df.sum(axis=1)


# ===================================================================================
#####################################################################################
# From the web page challenge
#####################################################################################
# ===================================================================================
# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

speed = np.random.choice(np.arange(200))
light_count = np.random.choice(np.arange(9999))

# speed = int(input())
# light_count = int(input())
for i in range(light_count):
    distance, duration = np.random.choice(np.arange(99999)), np.random.choice(np.arange(9999))

# Write an action using print
# To debug: print("Debug messages...", file=sys.stderr)

print("answer")
