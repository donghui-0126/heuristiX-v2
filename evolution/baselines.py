"""Baseline dispatching rules expressed in the simulator's evalexpr DSL.

These map 1:1 to the Rust `baselines::*` rules but are written as DSL
strings so the same evolution machinery can mutate, combine, and refine
them. The simulator's CLI accepts them via `--rule expr --expr "<dsl>"`.

Convention: higher score = higher priority. To get "earliest X first"
behavior we negate X.
"""

# Each value is a single evalexpr expression returning a Float.
# Bindings exposed: see src/rules/expr.rs ExprRule docstring.
#
# The first five are the report's H1–H5 (창종설 §4-1). The last three
# come from the dispatching-rule literature and are commonly used as
# stronger baselines in JSSP papers — included here for benchmarking the
# evolution loop against rules that already encode tardiness-cost trade-
# offs by hand.
BASELINES: dict[str, str] = {
    "FIFO":    "0.0 - release",
    "EDD":     "0.0 - due",
    "SPT":     "0.0 - proc",
    # CR: smaller (due - now) / remaining_proc → higher priority.
    "CR":      "0.0 - (due - now) / max_(remaining_proc, 0.001)",
    "Urgency": "iff(urgent, 1.0, 0.0)",

    # WMDD (Weighted Modified Due Date, Eilon-Chowdhury 1976):
    # selection rule: pick min { max(d_j, t + p_j) / w_j }.
    # Higher weight (penalty) AND smaller modified due-date → higher priority.
    "WMDD":    "0.0 - max_(due, now + proc) / max_(penalty, 0.001)",

    # COVERT (Cost OVER Time, Carroll 1965):
    # u = max(0, slack - proc) / (k * proc)
    # cover = max(0, 1 - u)
    # priority = (penalty / proc) * cover     (k = 2 here)
    "COVERT":  "(penalty / max_(proc, 0.001)) * "
               "max_(0.0, 1.0 - max_(0.0, (due - now) - proc) / (2.0 * max_(proc, 0.001)))",

    # ATC (Apparent Tardiness Cost, Vepsalainen-Morton 1987):
    # priority = (penalty / proc)
    #          * exp(-max(0, slack - proc) / (k * proc))    (k = 3)
    "ATC":     "(penalty / max_(proc, 0.001)) * "
               "exp_(0.0 - max_(0.0, (due - now) - proc) / (3.0 * max_(proc, 0.001)))",

    # LPT (Longest Processing Time first). Bad on tardiness but useful as
    # a load-balancing contrast. Holthaus & Rajendran 2000.
    "LPT":     "proc",

    # MWKR (Most Work Remaining first). Classic JSSP — Conway-Maxwell-Miller.
    # Helps reduce makespan by pushing long jobs through early.
    "MWKR":    "remaining_proc",

    # LWKR (Least Work Remaining first) = SRPT in our JSSP setup.
    # Mirror of MWKR; reduces mean flow time.
    "LWKR":    "0.0 - remaining_proc",

    # MDD (Modified Due Date, Baker-Bertrand 1982), unweighted version of
    # WMDD. Selection: min { max(d_j, t + p_j) }.
    "MDD":     "0.0 - max_(due, now + proc)",
}


def baseline_names() -> list[str]:
    return list(BASELINES.keys())
