from math import isnan, nan
from typing import Callable, Tuple

from numpy import ndarray

from ..jit import gjit, array_to_V_hf
from ..math.ieee754 import EPS
from ..math.ivp import (
    BRENTQ_CONVERGED,
    BRENTQ_MAXITER,
    DENSE_SIG,
    DOP853_FINISHED,
    DOP853_FAILED,
    DOP853_ARGK,
    DOP853_FR,
    DOP853_FUN,
    DOP853_FV,
    DOP853_H_PREVIOUS,
    DOP853_K,
    DOP853_RR,
    DOP853_RR_OLD,
    DOP853_STATUS,
    DOP853_T,
    DOP853_T_OLD,
    DOP853_VV,
    DOP853_VV_OLD,
    brentq_dense_hf,
    event_is_active_hf,
    dispatcher_hb,
    dop853_dense_interp_hf,
    dop853_dense_output_hf,
    dop853_init_hf,
    dop853_step_hf,
)
from ..propagation.base import func_twobody_hf


__all__ = [
    "cowell_vb",
    "SOLVE_BRENTQFAILED",
    "SOLVE_FAILED",
    "SOLVE_RUNNING",
    "SOLVE_FINISHED",
    "SOLVE_TERMINATED",
]


SOLVE_BRENTQFAILED = -3
SOLVE_FAILED = -2
SOLVE_RUNNING = -1
SOLVE_FINISHED = 0
SOLVE_TERMINATED = 1


def cowell_vb(
    events: Tuple = tuple(),
    func: Callable = func_twobody_hf,
) -> Callable:
    """
    Builds vectorized cowell
    """

    assert hasattr(func, "djit")  # DEBUG check for compiler flag

    EVENTS = len(events)  # TODO compile as const

    event_impl_hf = dispatcher_hb(
        funcs=tuple(event.impl_hf for event in events),
        argtypes="f,V,V,f",
        restype="f",
        arguments="t, rr, vv, k",
    )
    event_impl_dense_hf = dispatcher_hb(
        funcs=tuple(event.impl_dense_hf for event in events),
        argtypes=f"f,{DENSE_SIG:s},f",
        restype="f",
        arguments="t, t_old, h, rr_old, vv_old, F, argk",
    )

    @gjit(
        "void(f[:],f[:],f[:],f,f,f,b1[:],f[:],f[:],f[:],f[:],f[:],i8[:],i8[:],f[:,:],f[:,:])",
        "(n),(m),(m),(),(),(),(o),(o)->(o),(o),(o),(o),(),(),(n,m),(n,m)",
        cache=False,
    )  # n: tofs, m: dims, o: events
    def cowell_gf(
        tofs: ndarray,
        rr: Tuple[float, float, float],
        vv: Tuple[float, float, float],
        argk: float,
        rtol: float,
        atol: float,
        event_terminals: ndarray,
        event_directions: ndarray,
        event_g_olds: ndarray,  # (out)
        event_g_news: ndarray,  # (out)
        event_actives: ndarray,  # out
        event_last_ts: ndarray,  # out
        status: int,  # out
        t_idx: int,  # out
        rrs: ndarray,  # out
        vvs: ndarray,  # out
    ):  # -> void(..., rrs, vvs, success)
        """
        Solve an initial value problem for a system of ODEs.

        Can theoretically be reversed: https://github.com/poliastro/poliastro/issues/1630
        """

        # assert isinstance(rtol, float)
        # assert all(tof >= 0 for tof in tofs)
        # assert sorted(tofs) == list(tofs)

        T0 = 0.0

        solver = dop853_init_hf(
            func, T0, array_to_V_hf(rr), array_to_V_hf(vv), tofs[-1], argk, rtol, atol
        )

        t_idx[0] = 0
        t_last = T0

        for event_idx in range(EVENTS):
            event_g_olds[event_idx] = event_impl_hf(
                event_idx, T0, array_to_V_hf(rr), array_to_V_hf(vv), argk
            )
            event_last_ts[event_idx] = T0

        status[0] = SOLVE_RUNNING
        while status[0] == SOLVE_RUNNING:
            solver = dop853_step_hf(*solver)

            if solver[DOP853_STATUS] == DOP853_FINISHED:
                status[0] = SOLVE_FINISHED
            elif solver[DOP853_STATUS] == DOP853_FAILED:
                status[0] = SOLVE_FAILED
                break

            t_old = solver[DOP853_T_OLD]
            t = solver[DOP853_T]

            interpolant = dop853_dense_output_hf(
                solver[DOP853_FUN],
                solver[DOP853_ARGK],
                solver[DOP853_T_OLD],
                solver[DOP853_T],
                solver[DOP853_H_PREVIOUS],
                solver[DOP853_RR],
                solver[DOP853_VV],
                solver[DOP853_RR_OLD],
                solver[DOP853_VV_OLD],
                solver[DOP853_FR],
                solver[DOP853_FV],
                solver[DOP853_K],
            )

            at_least_one_active = False
            for event_idx in range(EVENTS):
                event_g_news[event_idx] = event_impl_hf(
                    event_idx, t, solver[DOP853_RR], solver[DOP853_VV], argk
                )
                event_last_ts[event_idx] = t
                event_actives[event_idx] = event_is_active_hf(
                    event_g_olds[event_idx],
                    event_g_news[event_idx],
                    event_directions[event_idx],
                )
                if event_actives[event_idx]:
                    at_least_one_active = True

            if at_least_one_active:
                root_pivot = nan  # set initial value
                terminate = False

                for event_idx in range(EVENTS):
                    if not event_actives[event_idx]:
                        continue

                    if not event_terminals[event_idx]:
                        continue

                    terminate = True

                    event_last_ts[event_idx], root, brentq_status = brentq_dense_hf(
                        event_impl_dense_hf,
                        event_idx,
                        t_old,
                        t,
                        4 * EPS,
                        4 * EPS,
                        BRENTQ_MAXITER,
                        *interpolant,
                        argk,
                    )
                    if brentq_status != BRENTQ_CONVERGED:
                        status[0] = SOLVE_BRENTQFAILED
                        return  # failed on event

                    if isnan(root_pivot):
                        root_pivot = root
                        continue

                    if t > t_old:  # smallest root of all active events
                        if root < root_pivot:
                            root_pivot = root
                        continue

                    # largest root of all active events
                    if root > root_pivot:
                        root_pivot = root
                    raise ValueError("not t > t_old", t, t_old)  # TODO remove

                if terminate:
                    assert not isnan(root_pivot)
                    status[0] = SOLVE_TERMINATED
                    t = root_pivot

            for event_idx in range(EVENTS):
                event_g_olds[event_idx] = event_g_news[event_idx]

            if not t_last <= t:
                raise ValueError("not t_last <= t", t_last, t)

            while t_idx[0] < tofs.shape[0] and tofs[t_idx[0]] < t:
                rrs[t_idx[0], :], vvs[t_idx[0], :] = dop853_dense_interp_hf(
                    tofs[t_idx[0]], *interpolant
                )
                t_idx[0] += 1
            if status[0] == SOLVE_TERMINATED or tofs[t_idx[0]] == t:
                rrs[t_idx[0], :], vvs[t_idx[0], :] = dop853_dense_interp_hf(
                    t, *interpolant
                )
                t_idx[0] += 1

            t_last = t

    return cowell_gf
