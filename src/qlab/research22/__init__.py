"""Research 22: evolutionary (GP) search for equity selection rules, evaluated against random search and nulls.

This package holds the *evaluator*: a compacted weekly panel, a small typed expression grammar, numba kernels
for ranking / top-k selection / costs and the fold-quantile fitness. The search driver (DEAP), the nulls and the
meta-analysis come later; nothing here selects strategies on its own.

Contract (same as the rest of qlab): row t of every terminal uses data up to the close of day t; the decision
is taken after the close of the first trading day of the week (`qlab.schedule.period_starts`) and executed at
the next day's open.
"""
