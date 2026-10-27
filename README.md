# ABM RGDU — the RGDU, technology and employment

A small agent-based model built to answer one question: under which conditions does a cut in the RGDU, the general reduction of employer contributions in force since 2026, change employment, through the cost of labor, through automation, and through demand?

The model does not answer it for France. It answers it for an experimental sample of 120 firms whose rules are explicit and whose parameters are, for the most part, hypotheses. What it can say is which way employment moves in that sample, through which channel, and how far that depends on things nobody has measured yet. Every output carries the label **EXPERIMENTAL — parameters not identified**, and it should be read that way.

## What is in the model

**Firms.** 120 of them, in four markets (manufacturing, trade, accommodation and food, business services), three sizes (2–9, 10–49 and 50–250 posts) and ten replications. They all follow the same rules, from one `Firm` class; only their states differ. Heterogeneity is drawn conditionally on market and size, never written firm by firm.

**Two tasks.** Production needs routine work (A) and other work (B), in fixed proportions: q = min(demand, (L_A + M)/a, L_B/b). A machine adds A capacity, but it can never do more than the automatable share θ of A work. The rest needs people, however cheap the machine.

**Wages and the RGDU.** Within each task, posts fall into four wage bands, from 1 to 3 SMIC and above. The reduction is computed post by post, band by band, then summed. Because the reduction is degressive and stops dead at 3 SMIC, applying it to an average wage gets it wrong, and badly so: −43.8% on the example of the specification. The reference SMIC is the one of 1 January 2026, frozen by décret 2026-509. The current SMIC, 12.31 € an hour since June, only checks that wages are legal.

**Staffing.** A lone A post adds nothing while B binds, so a firm does not weigh its posts one by one. It compares output levels, from zero up to expected demand, each with the bundle of A and B posts it takes, over a short horizon and with real dates.
- Hires work from the next quarter.
- Laid-off workers stay one more quarter, paid, before leaving with severance.
- Quits that are not replaced cost nothing, so a shrinking firm shrinks by attrition first.
- Nothing obliges a firm to keep an activity that loses money.

**Technology.** Each firm has one option: a machine sized to the automatable share of its A work, priced per post of capacity, independently of its wages. Two questions are kept apart.
- *Is it worth it?* An NPV over sixteen quarters compares two projections made by the same planner, with no financing flow in it.
- *Can it be paid for?* Cash above a buffer first, then a loan with interest and installments, under one credit ceiling.

A profitable machine can therefore be refused for want of financing. A machine fires nobody by construction. With rigid prices the adopter needs fewer people for the same sales. With price-sensitive customers it may sell enough to employ more. Both cases are in the tests.

**Markets.** Each market has a budget, shared out by inherited share and relative price. Prices follow costs with a lag. Demand a firm cannot serve is lost, unless a share of it is offered to rivals with spare capacity (off in the reference). In one variant, budgets follow last quarter's labor income and ARE.

**Labor pools.** Two pools, by competence, in a closed field: employment plus unemployment is the active population at every quarter. The unemployed are counted by duration, which is what the ARE proxy needs; it is a proxy, not an account of rights.

## How effects are measured

For each seed, the model builds one world, runs a common burn-in at α = 1 with no investment, then starts every branch from that same state with the same random draws. α is the share of the RGDU that is kept: 1 is the reference, 0.9 and 0.8 are cuts of 10% and 20%, and 0 is a separate stress test.

    ΔE_tech     = E(α, technology possible)   − E(1, technology possible)
    ΔE_without  = E(α, technology impossible) − E(1, technology impossible)
    interaction = ΔE_tech − ΔE_without

The reference may invest too. Only adoptions that are additional or earlier under the reform count as responses to it.

The *partial public balance* (employer contributions net of the RGDU, minus the ARE proxy) is split into two parts:
- the mechanical gain of the cut, computed on the reference's employment;
- what the change in employment takes back.

It leaves out employee contributions, CSG, taxes and most benefits, so it says nothing about public finances as a whole.

Two kinds of spread are reported, never pooled:
- the spread across seeds, which reflects random draws and nothing else;
- the spread across the parameter grid, which reflects the hypotheses.

A Monte-Carlo interval says nothing about whether a structural assumption is right.

If the sign of ΔE turns with a parameter that is not identified, the honest conclusion is a frontier of conditions, not a number.

## Running it

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    .venv/bin/pytest
    .venv/bin/python -m abm_rgdu.experiments --seeds 10 --out results/ --figures
    .venv/bin/python -m abm_rgdu.experiments --seeds 10 --out results/ --grid --figures   # adds the hypothesis grid

A seed takes about a minute of one core, mostly spent on investment projections. Seeds run in parallel (`--workers`). The grid is several times longer.

## Changing parameters and time

Every parameter sits in `data/params_v1_5.json`, under the name you see there. It can be changed in two ways, from the lightest to the most lasting.

- **For one run, on the command line.** Name the parameter by its section and key. Values are read as JSON, so numbers, lists and `true`/`false` work as written:

      .venv/bin/python -m abm_rgdu.experiments --seeds 4 --set demand.eta=5 --set technology.unit_cost=60000
      .venv/bin/python -m abm_rgdu.experiments --quarters 12 --burn-in 0
      .venv/bin/python -m abm_rgdu.experiments --set "markets.automatable_mean=[0.6, 0.5, 0.3, 0.5]"

  `--quarters` is the horizon after the reform; `--burn-in` is the number of shared quarters at α = 1 before it. A misspelt name is refused rather than silently ignored.
- **In a copy of the file**, passed with `--params my_params.json`. This is the right place for a set of hypotheses you mean to keep, with its notes and statuses.

Outputs are reported at 2, 4 and 8 quarters, or at whatever `experiments.report_at` says, within the horizon; the last quarter is always included.

## What comes out

Every file opens with the status of its parameters.

| File | Content |
|---|---|
| `registry.csv` | every parameter family with its status (R rule, E checked estimate, H hypothesis) and its source |
| `levels.csv` | indicators at 2, 4 and 8 quarters, by branch, market and size |
| `paths.csv` | the same for the whole sample, quarter by quarter |
| `effects.csv`, `effects_summary.csv` | paired gaps, seed by seed, then mean and spread across seeds |
| `drift.csv` | why the reference moves: vacancies, hires, quits, layoffs, non-replacements |
| `validation.csv` | comparison with held-out observations; "not verified" until they are checked |
| `grid.csv`, `competition_income.csv` | effects over unidentified parameters (`--grid`) |
| `figures/` | employment, adoptions, the cost frontier by band, competition and income, the balance, sign maps |

Totals are totals of a sample that is uniform over twelve cells, and so not representative. Results are meant to be read by market and by size.

## Parameters and their status

All parameters live in `data/params_v1_5.json`, each family with a status, a source and a note.

- **R (official rules):**
  - the RGDU 2026 schedule (Tmin, Tdelta, exponent, 3 SMIC);
  - the frozen reference SMIC;
  - 2026 employer rates;
  - the 0.49% cap on the eligible AT/MP part.
- **E (estimates checked against a source):** the VA/CA ratio of each market, from INSEE Ésane 2023. "Industrie" stands in for manufacturing.
- **H (hypotheses):** everything else, until checked:
  - margins and cash by size;
  - band wages, task shares and automatable shares;
  - technology costs;
  - price sensitivity and pass-through;
  - quits and matching;
  - the ARE proxy and credit ceilings.

  Six families matter most and are the ones to sweep: automatable share, technology cost and performance, price sensitivity of customers, pass-through of costs to prices, access to finance, and re-employment. Income feedback is a seventh, optional.

`data/validation.json` holds observations kept apart from calibration. None is verified yet.

## Known limits

- **The sample.**
  - The 120 firms are an experimental sample.
  - Large firms are stylized at 50 to 250 posts, not national groups.
  - Crossing size and market for margins is an assumption.
- **Pricing.** Prices are a markup over full unit cost, adjusted with a lag. With fixed budget shares and fast pass-through, a firm hands its automation savings to its customers and has little reason to invest. That is a property of the rule, not of the world, and worth testing against pass-through (λ) and price sensitivity (η).
- **Labor.**
  - There is no substitution between wage bands within a task: hires and layoffs keep each firm's band mix.
  - No mobility from pool A to pool B, which makes displaced routine workers depend on demand for routine work.
  - Full-time posts only, so FTE equals persons.
  - The FNAL profile is fixed at the start.
- **Payroll.** Annual amounts are divided by four. Annual regularisations are not reproduced.
- **Investment.** Rivals' prices are held fixed and falling technology costs are not anticipated. Projections assume vacancies are filled and quits take their expected value.
- **The reference drifts slightly.** Some vacancies go unfilled each quarter, and a post barely used is not replaced after a quit. The drift report shows it; it is not hidden by tuning.
- **Robustness switches.** Closures, negotiated wages, credit regimes, technology cost paths and an initial stock of machines exist as switches and are off by default. They are not to be extrapolated.

## Sources

- S1 — Ministère de l'Économie, RGDU 2026: https://www.economie.gouv.fr/entreprises/gerer-ses-ressources-humaines-et-ses-salaries/comment-fonctionne-la-reduction-generale-degressive-unique-rgdu-de-cotisations-patronales
- S2 — INSEE, Panorama de l'appareil productif en 2023: https://www.insee.fr/fr/statistiques/8679855
- S3 — INSEE, Ratios des secteurs marchands par activité en 2023: https://www.insee.fr/fr/statistiques/2015421
- S4 — INSEE, Salaires dans le secteur privé, édition 2026: https://www.insee.fr/fr/statistiques/8733089
- S5 — Dares, Mouvements de main-d'œuvre: https://dares.travail-emploi.gouv.fr/donnees/les-mouvements-de-main-doeuvre
- S6 — Acemoglu, Lelarge & Restrepo (2020), robots and firms in France: https://www.aeaweb.org/articles?id=10.1257%2Fpandp.20201003
- Décret 2026-509 of 12 June 2026 (reference SMIC frozen at 12.02 €), arrêté of 22 May 2026 (SMIC at 12.31 € from 1 June 2026).

Technology prices and the H parameters come from none of these: they are scenario variations. No series 2000–2016 rebased to 2020 is used to calibrate technological capital.

## License

Apache-2.0, see `LICENSE`.
