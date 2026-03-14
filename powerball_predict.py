import pandas as pd, numpy as np, urllib.request
from io import StringIO
from itertools import combinations
from collections import Counter
import datetime

print("📥 Downloading official Powerball data ...")

# Current working Powerball dataset from data.ny.gov
url = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
csv_data = urllib.request.urlopen(url).read().decode("utf-8")
df = pd.read_csv(StringIO(csv_data))

# Normalize column names
df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
if not {"drawdate","winningnumbers"}.issubset(df.columns):
    print("DEBUG: Columns detected →", df.columns.tolist())
    raise KeyError("⚠️ Could not locate expected columns in Powerball dataset.")

df = df.dropna(subset=["drawdate","winningnumbers"])
df = df.sort_values("drawdate")

# --- Recency decay weighting ---
decay = 0.995
weights = decay ** np.arange(len(df))[::-1]

white_counts = np.zeros(69)
red_counts   = np.zeros(26)
pair_counter = Counter()

# --- Parse and tally draws (6 numbers per line) ---
for i, row in enumerate(df["winningnumbers"]):
    try:
        nums = [int(x) for x in str(row).split()]
        if len(nums) != 6:  # Expecting 5 white + 1 Powerball
            continue
    except Exception:
        continue
    whites, red = nums[:5], nums[5]
    w = weights[i]
    for n in whites:
        if 1 <= n <= 69:
            white_counts[n-1] += w
    if 1 <= red <= 26:
        red_counts[red-1] += w
    for a,b in combinations(sorted(whites),2):
        pair_counter[(a,b)] += w

# --- Laplace smoothing ---
white_prob = (white_counts + 1) / (white_counts.sum() + 69)
red_prob   = (red_counts   + 1) / (red_counts.sum()   + 26)

def draw_one():
    whites = np.sort(np.random.choice(np.arange(1,70),5,replace=False,p=white_prob))
    red = np.random.choice(np.arange(1,27),1,p=red_prob)[0]
    freq_score = white_prob[whites-1].sum() + red_prob[red-1]
    pair_score = sum(pair_counter.get(tuple(sorted(p)),0) for p in combinations(whites,2))
    return (tuple(whites)+(int(red),), freq_score + 0.0001*pair_score)

print("🔄 Bootstrapping 40000 Powerball draws ...")
samples = Counter()
for _ in range(40000):
    combo, score = draw_one()
    samples[combo] += score

top5 = samples.most_common(5)
print("\n🎯 Top 5 Powerball Predictions (stability ranked):")
for i,(combo,score) in enumerate(top5,1):
    print(f"{i}. {list(combo)} (stability score={score:.2f})")

# --- Save results with timestamp ---
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = f"powerball_top5_{timestamp}.csv"
pd.DataFrame([list(c) for c,_ in top5]).to_csv(filename, index=False)
print(f"\n💾 Results saved → {filename}\n")

input("Press Enter to exit ...")

