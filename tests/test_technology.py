import copy

import numpy as np

from abm_rgdu import firms
from abm_rgdu import simulator as sim
from abm_rgdu.firms import A, Machine, requirements
from worlds import small_world

ADJ = (0.5, 1.25)
CHEAP = {"technology.unit_cost": 5000.0, "markets.automatable_mean": [0.6]}
# Slow pass-through: the adopter keeps its savings long enough for the machine to pay.
ADOPT = {"technology.unit_cost": 20000.0, "demand.lambda": 0.1}


def warmed(p, seed=5):
    state = sim.initial_state(p, seed)
    sim.burn_in(state)
    return state


def test_machine_adds_nothing_before_it_is_installed():
    f = copy.deepcopy(warmed(small_world()).firms[0])
    f.machine = Machine(capacity=10.0, cost=1.0, maintenance=0.1, transition=0.2, ordered_at=3, active_from=5)
    assert f.machine_at(3) == f.machine_at(4) == 0.0
    assert f.machine_at(5) == 10.0
    assert f.maintenance_at(4) == 0.0 and f.maintenance_at(5) == 0.1


def test_a_posts_are_cut_only_once_the_machine_runs():
    f = copy.deepcopy(warmed(small_world()).firms[0])
    m, demand = f.machine_capacity, f.initial_output
    margin = 10 * f.amount("cost") / demand
    # Active from t+2: notices go out now, but staff in t+1 is still sized without it.
    plan = firms.plan_staff(f, demand, margin, [0.0, m, m, m], ADJ)
    without = requirements(plan.output, 0.0, f.a, f.b, f.theta)
    with_machine = requirements(plan.output, m, f.a, f.b, f.theta)
    assert plan.next_staff[A] >= without[A]
    assert plan.target[A] == with_machine[A] < without[A]
    assert plan.notify[A] > 0
    # Active only from t+3: nothing is cut yet.
    plan = firms.plan_staff(f, demand, margin, [0.0, 0.0, m, m], ADJ)
    assert plan.notify[A] == 0
    assert plan.target[A] == requirements(plan.output, 0.0, f.a, f.b, f.theta)[A]


def test_capacity_and_automatable_share_hold_every_quarter():
    state = warmed(small_world(**CHEAP))
    rec = sim.run(copy.deepcopy(state), sim.Branch(0.8, True))
    assert rec.get("firm", "ordered").sum() > 0
    theta = np.array([f.theta for f in state.firms])
    a = np.array([f.a for f in state.firms])
    output = rec.get("firm", "output")
    staff_a = rec.get("firm", "employment")[:, :, A]
    assert np.all((1 - theta) * a * output <= staff_a + 1e-9)
    assert np.all(output <= rec.get("firm", "capacity") + 1e-9)


def test_adoption_can_happen_in_the_reference():
    state = warmed(small_world(**ADOPT))
    on = sim.run(copy.deepcopy(state), sim.Branch(1.0, True))
    off = sim.run(copy.deepcopy(state), sim.Branch(1.0, False))
    assert on.get("firm", "ordered").sum() > 0
    assert off.get("firm", "ordered").sum() == 0
    assert np.all(off.get("firm", "machine_capacity") == 0)


def test_technology_costs_are_the_same_in_reference_and_reform():
    state = warmed(small_world(**ADOPT))
    ref, pol = copy.deepcopy(state), copy.deepcopy(state)
    sim.run(ref, sim.Branch(1.0, True))
    sim.run(pol, sim.Branch(0.8, True))
    pairs = [(r.machine, q.machine) for r, q in zip(ref.firms, pol.firms) if r.machine and q.machine]
    assert pairs
    for r, q in pairs:
        assert (r.cost, r.maintenance, r.transition, r.capacity) == (q.cost, q.maintenance, q.transition, q.capacity)


def test_both_employment_signs_are_reachable():
    """A machine fires nobody by construction. With rigid prices the adopter needs
    fewer people for the same sales; with pass-through and price-sensitive customers
    it sells so much more that it employs more. The sign is not written in the rules."""
    signs = []
    for eta, lam in ((2.0, 0.0), (8.0, 0.5)):
        state = warmed(small_world(**CHEAP, **{"demand.eta": eta, "demand.lambda": lam}))
        for f in state.firms[1:]:
            f.machine_capacity = 0.0  # a single possible adopter
        on = sim.run(copy.deepcopy(state), sim.Branch(1.0, True))
        off = sim.run(copy.deepcopy(state), sim.Branch(1.0, False))
        assert on.get("firm", "ordered")[:, 0].any()
        gap = on.get("firm", "employment")[-1, 0].sum() - off.get("firm", "employment")[-1, 0].sum()
        signs.append(int(np.sign(gap)))
    assert signs == [-1, 1]


def test_the_reform_can_tip_an_investment_the_reference_would_not_make():
    state = warmed(small_world(**{"technology.unit_cost": 20000.0, "demand.lambda": 0.5}))
    ref = sim.run(copy.deepcopy(state), sim.Branch(1.0, True))
    pol = sim.run(copy.deepcopy(state), sim.Branch(0.8, True))
    assert ref.get("firm", "ordered").sum() < pol.get("firm", "ordered").sum()
