"""
DLL 単体スモークテスト:
  - ssd_create / ssd_step / ssd_get_kappa_row / ssd_set_params / ssd_get_params の疎通確認
実行: python test_ssd_dll_smoke.py
"""

import ctypes
import os

DLL_PATH = "./ssd_align_leap.dll"

class SSDParams(ctypes.Structure):
    _fields_ = [
        ("G0", ctypes.c_double), ("g", ctypes.c_double), ("eps_noise", ctypes.c_double),
        ("eta", ctypes.c_double), ("rho", ctypes.c_double), ("lam", ctypes.c_double),
        ("kappa_min", ctypes.c_double), ("alpha", ctypes.c_double), ("beta_E", ctypes.c_double),
        ("Theta0", ctypes.c_double), ("a1", ctypes.c_double), ("a2", ctypes.c_double),
        ("h0", ctypes.c_double), ("gamma", ctypes.c_double), ("T0", ctypes.c_double),
        ("c1", ctypes.c_double), ("c2", ctypes.c_double), ("sigma", ctypes.c_double),
        ("delta_w", ctypes.c_double), ("delta_kappa", ctypes.c_double), ("c0_cool", ctypes.c_double),
        ("q_relax", ctypes.c_double), ("eps_relax", ctypes.c_double), ("eps0", ctypes.c_double),
        ("d1", ctypes.c_double), ("d2", ctypes.c_double), ("b_path", ctypes.c_double),
    ]


class SSDTelemetry(ctypes.Structure):
    _fields_ = [
        ("E", ctypes.c_double), ("Theta", ctypes.c_double), ("h", ctypes.c_double),
        ("T", ctypes.c_double), ("H", ctypes.c_double), ("J_norm", ctypes.c_double),
        ("align_eff", ctypes.c_double), ("kappa_mean", ctypes.c_double),
        ("current", ctypes.c_int32), ("did_jump", ctypes.c_int32), ("rewired_to", ctypes.c_int32),
    ]


def main():
    if not os.path.exists(DLL_PATH):
        raise SystemExit(f"DLL not found: {DLL_PATH}")
    dll = ctypes.CDLL(DLL_PATH)

    dll.ssd_create.argtypes = [ctypes.c_int32, ctypes.POINTER(SSDParams), ctypes.c_uint64]
    dll.ssd_create.restype = ctypes.c_void_p
    dll.ssd_destroy.argtypes = [ctypes.c_void_p]
    dll.ssd_step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.POINTER(SSDTelemetry)]
    dll.ssd_get_kappa_row.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
    dll.ssd_get_kappa_row.restype = ctypes.c_int32
    dll.ssd_get_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(SSDParams)]
    dll.ssd_set_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(SSDParams)]
    dll.ssd_get_N.argtypes = [ctypes.c_void_p]
    dll.ssd_get_N.restype = ctypes.c_int32

    params = SSDParams()
    # デフォルト類似初期化
    params.G0=0.5; params.g=0.7; params.eps_noise=0.0; params.eta=0.3; params.rho=0.3; params.lam=0.02; params.kappa_min=0.0
    params.alpha=0.6; params.beta_E=0.15
    params.Theta0=1.0; params.a1=0.5; params.a2=0.4; params.h0=0.2; params.gamma=0.8
    params.T0=0.3; params.c1=0.5; params.c2=0.6
    params.sigma=0.2
    params.delta_w=0.2; params.delta_kappa=0.2; params.c0_cool=0.6; params.q_relax=0.1; params.eps_relax=0.01
    params.eps0=0.02; params.d1=0.2; params.d2=0.2; params.b_path=0.5

    handle = dll.ssd_create(8, ctypes.byref(params), 123456789)
    if not handle:
        raise SystemExit("ssd_create failed")

    print("Created SSD handle.")

    tele = SSDTelemetry()
    for step, mp in enumerate([0.05, 0.2, 0.6, 1.2, 1.8], 1):
        dll.ssd_step(handle, mp, 1.0, ctypes.byref(tele))
        p_jump = 1.0 - pow(2.718281828, -tele.h)  # approx exp(-h)
        print(f"[{step}] p={mp:.2f} node={tele.current} jump={tele.did_jump} "
              f"E={tele.E:.4f} h={tele.h:.4f} p_jump≈{p_jump:.3f} kappa_mean={tele.kappa_mean:.5f}")

    # kappa row check
    buf = (ctypes.c_double * 8)()
    count = dll.ssd_get_kappa_row(handle, 0, buf, 8)
    row_vals = [buf[i] for i in range(count)]
    print("kappa row 0:", row_vals)

    # get/set params round trip
    p2 = SSDParams()
    dll.ssd_get_params(handle, ctypes.byref(p2))
    print("Fetched param G0:", p2.G0)
    p2.G0 = 0.55
    dll.ssd_set_params(handle, ctypes.byref(p2))
    p3 = SSDParams()
    dll.ssd_get_params(handle, ctypes.byref(p3))
    print("Updated param G0:", p3.G0)

    dll.ssd_destroy(handle)
    print("Destroyed SSD handle. Smoke test OK.")


if __name__ == "__main__":
    main()