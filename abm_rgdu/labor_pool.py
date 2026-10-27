"""Two labor pools, by competence: A for routine tasks, B for the others.

The field is closed: whoever leaves a firm joins the pool of their task and
whoever is hired comes out of it, so employment plus unemployment equals the
active population at every quarter. There is no Person object. The unemployed
are counted by completed quarters of unemployment, which is all the ARE proxy
needs; it is a proxy, not a reckoning of individual rights.
"""

from dataclasses import dataclass

import numpy as np

from .firms import A, B

SLOTS = 6                          # completed quarters of unemployment: 0, 1, 2, 3, 4, 5+
BUCKET = np.array([0, 0, 1, 1, 1, 2])  # ARE buckets: 0–1, 2–4, 5+ quarters


def apportion(total, weights, caps, tiebreak=None):
    """Whole numbers summing to `total` (or to the caps), proportional to `weights`.

    Largest remainders get the leftover units, ties broken by `tiebreak`;
    nobody receives more than its cap.
    """
    weights = np.asarray(weights, dtype=float)
    caps = np.asarray(caps, dtype=int)
    tiebreak = np.zeros(len(weights)) if tiebreak is None else np.asarray(tiebreak, dtype=float)
    out = np.zeros(len(weights), dtype=int)
    left = int(min(total, caps.sum()))
    while left > 0:
        room = caps - out
        w = np.where(room > 0, weights, 0.0)
        if w.sum() <= 0:
            w = (room > 0).astype(float)
        quota = left * w / w.sum()
        give = np.minimum(np.floor(quota + 1e-9).astype(int), room)
        out += give
        left -= int(give.sum())
        room = caps - out
        for i in np.lexsort((tiebreak, -(quota - give))):
            if left == 0:
                break
            if room[i] > 0 and w[i] > 0:
                out[i] += 1
                left -= 1
    return out


def stochastic_round(x, u):
    """floor(x + u) with u uniform: unbiased, and shared draws keep branches comparable."""
    return np.floor(np.asarray(x, dtype=float) + u).astype(int)


@dataclass
class LaborPool:
    task: int
    active: int             # active population of the field for this competence
    slots: np.ndarray       # unemployed by completed quarters, last slot is 5+
    reference_wage: float   # annual gross, for the ARE proxy only
    incoming: int = 0       # separations of the quarter, unemployed from the next one

    def unemployed(self):
        """Unemployed during the current quarter."""
        return int(self.slots.sum())

    def available(self):
        return int(self.slots.sum()) + self.incoming

    def by_bucket(self):
        return np.bincount(BUCKET, weights=self.slots, minlength=3)

    def receive(self, n):
        self.incoming += int(n)

    def hire(self, n, weights):
        """Take n people out of the pool, more often among the recently unemployed.

        Those separated this very quarter count as zero completed quarters.
        """
        counts = np.concatenate([[self.incoming], self.slots])
        w = np.concatenate([[weights[0]], np.asarray(weights)[BUCKET]]) * counts
        taken = apportion(n, w, counts)
        self.incoming -= int(taken[0])
        self.slots = self.slots - taken[1:]
        return int(taken.sum())

    def age(self):
        """End of quarter: everyone still unemployed has one more quarter behind them."""
        nxt = np.zeros(SLOTS, dtype=int)
        nxt[0] = self.incoming
        nxt[1:SLOTS - 1] = self.slots[:SLOTS - 2]
        nxt[SLOTS - 1] = self.slots[SLOTS - 2] + self.slots[SLOTS - 1]
        self.slots = nxt
        self.incoming = 0

    def are(self, p):
        """Quarterly ARE proxy: share compensated × typical amount, by duration bucket."""
        a = p["are"]
        monthly = a["replacement_rate"] * self.reference_wage / 12
        per_head = np.array(a["share_compensated"]) * np.array(a["amount_factor"]) * monthly * 3
        return float((self.by_bucket() * per_head).sum())


def build_pools(firms, p):
    lab = p["labor"]
    shares = np.array(lab["initial_duration_shares"], dtype=float)
    per_slot = shares[BUCKET] / np.bincount(BUCKET)[BUCKET]
    pools = []
    for task in (A, B):
        employed = int(sum(f.staff[task] for f in firms))
        active = int(round(employed / (1 - lab["initial_unemployment"][task])))
        unemployed = active - employed
        slots = apportion(unemployed, per_slot, np.full(SLOTS, unemployed))
        gross = sum(f.table("gross", task, int(f.staff[task]))[int(f.staff[task])] for f in firms) * 4
        pools.append(LaborPool(task, active, slots, gross / employed))
    return pools


def match(vacancies, prob, uniforms, available, tiebreak):
    """Vacancies filled with a common probability, rationed when the pool runs short."""
    vacancies = np.asarray(vacancies, dtype=int)
    filled = np.minimum(vacancies, stochastic_round(vacancies * prob, uniforms))
    if filled.sum() > available:
        filled = apportion(available, filled, filled, tiebreak)
    return filled
