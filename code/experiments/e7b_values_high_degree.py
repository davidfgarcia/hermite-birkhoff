"""E7b - values only with local degree 4 and 5 (wider Gaussian), for a fair comparison
with Hermite data at equal numbers of data.

Run from code/:  python3 -m experiments.e7b_values_high_degree
"""
import csv
from multiprocessing import Pool

import numpy as np

from experiments.e7_parameters import NS, run

JOBS = [(0, m, 2, c, n) for m in (4, 5) for c in (2.0, 3.0, 4.0, 5.0) for n in NS]

if __name__ == "__main__":
    with Pool(2) as pool:
        rows = pool.map(run, JOBS)
    with open("../results/e7b_values_high_degree.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for m in (4, 5):
        for c in (2.0, 3.0, 4.0, 5.0):
            rs = [r for r in rows if (r["degree"], r["c"]) == (m, c)]
            hs = np.array([r["h"] for r in rs]); es = np.array([r["max_err"] for r in rs])
            rm = np.array([r["rms_err"] for r in rs])
            print(f" 0  {m}  2  {c:3.1f}   " + " ".join(f"{e:.1e}" for e in es) +
                  f"   h^{np.polyfit(np.log(hs[-3:]), np.log(es[-3:]), 1)[0]:4.2f}"
                  f"      h^{np.polyfit(np.log(hs[-3:]), np.log(rm[-3:]), 1)[0]:4.2f}")
