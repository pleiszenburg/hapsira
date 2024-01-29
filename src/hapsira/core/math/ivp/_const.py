__all__ = [
    "N_RV",
    "N_STAGES",
    "SAFETY",
    "MIN_FACTOR",
    "MAX_FACTOR",
    "INTERPOLATOR_POWER",
    "N_STAGES_EXTENDED",
    "ERROR_ESTIMATOR_ORDER",
    "ERROR_EXPONENT",
    "KSIG",
]

N_RV = 6
N_STAGES = 12
N_STAGES_EXTENDED = 16

SAFETY = 0.9  # Multiply steps computed from asymptotic behaviour of errors by this.

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

INTERPOLATOR_POWER = 7
N_STAGES_EXTENDED = 16
ERROR_ESTIMATOR_ORDER = 7
ERROR_EXPONENT = -1 / (ERROR_ESTIMATOR_ORDER + 1)

KSIG = (
    "Tuple(["
    + ",".join(["Tuple([" + ",".join(["f"] * N_RV) + "])"] * (N_STAGES + 1))
    + "])"
)

FSIG = (
    "Tuple(["
    + ",".join(["Tuple([" + ",".join(["f"] * N_RV) + "])"] * INTERPOLATOR_POWER)
    + "])"
)
