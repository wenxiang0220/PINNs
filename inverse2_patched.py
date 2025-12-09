# -*- coding: utf-8 -*-
"""
PhysicsNeMo inverse fit — (R1 + R2, R3 off)
✅ 修正：RK4 使用 d/dx = DT * RHS（單位一致）
✅ 修正：R2 逆向= E^4
✅ 新增：loss 變化圖（artifacts/loss_curve.png）
✅ 新增：k1..k4, A0..E0 上下界依據 ODE 最佳化結果調整

物理模型（兩個進度變數 R, R2；秒為單位的 RHS）：
  R1:  Mg + H2  <->  MgH2                  (k1+, k1-)
  R2:  Mg17Al12 + 9 H2  <->  9 MgH2 + 4 Mg2Al3   (k3+, k4-)

  A = A0 - R,   C = C0 + R+9R2,
  D = D0 - R2,  E = E0 + 4 R2,
  H = B_init - R - 9 R2

ODE：
  dR/dt  = k1*(A)*H         - k2*(C)
  dR2/dt = k3*(D)*H**9      - k4*(C**9)*(E**4)
s
訓練在 x = (t - t0)/DT ∈ [0,1]，殘差使用：
  R'(x)  - DT * (dR/dt)  = 0
  R2'(x) - DT * (dR2/dt) = 0
"""

import sympy
import os, time, copy, glob
import numpy as np
import pandas as pd

from sympy import Symbol, Number, Function, exp
from physicsnemo.sym.eq.pde import PDE
import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseConstraint
from physicsnemo.sym.key import Key


@physicsnemo.sym.main(config_path="conf", config_name="config_primary_reaction")
def run(cfg: PhysicsNeMoConfig) -> None:
    # -------- helpers --------
    EPS = 1e-12
    def p(*a, **k):
        k.setdefault("flush", True)
        print(*a, **k)

    def find_torch_module(obj):
        try:
            import torch as _torch
        except Exception:
            return None
        for name in dir(obj):
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            if isinstance(v, _torch.nn.Module):
                return v
        return None

    def get_core_module(m):
        return m.module if hasattr(m, "module") else m

    def module_device(mod):
        try:
            for p_ in mod.parameters():
                return p_.device
            for b_ in mod.buffers():
                return b_.device
        except Exception:
            pass
        import torch
        return torch.device("cpu")

    # -------- data --------
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "dataset", "AZ61_3Pd_4574_8166sec.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # --- SLICE DATA TO 4574-8166s ---
    df = df[(df["Time_sec"] >= 4574) & (df["Time_sec"] <= 8166)].copy()

    # standardize column names
    df = df.rename(columns={
        "Time_sec": "Time_sec",
        "PT01_MPa": "pressure",
        "T01[Treac]_℃": "T_reac"
    })
    t = df["Time_sec"].to_numpy(dtype=float)
    idx = np.argsort(t)
    t = t[idx]
    P_MPa = df["pressure"].to_numpy(dtype=float)[idx]      # gauge MPa
    T_C   = df["T_reac"].to_numpy(dtype=float)[idx]

    # gauge→absolute, ideal gas → mol/L
    R_u = 8.31446261815324  # kPa·L/(mol·K)
    T_K = T_C + 273.15
    B_meas = (P_MPa + 0.101325)*1e3 / (R_u * T_K)  # mol/L

    t0, t1 = float(t.min()), float(t.max())
    DT = t1 - t0                              # 秒（訓練殘差要用）
    x  = (t - t0) / DT                        # 正規化時間 (0..1)

    # B_init: Use the first data point
    B_init = float(B_meas[0])
    p(f"[DATA] rows={len(t)},  t=[{t0:.1f},{t1:.1f}]  Δt={DT:.1f}s,  B_init≈{B_init:.6f} mol/L")

    # --- NO early weighting; 初始值=第一個資料點，且不複製早期資料 ---
    x_aug = x[:, None]
    B_aug = B_meas[:, None]
    zeros = np.zeros_like(x_aug)


    # -------- PARAM BOUNDS (from your ODE fit as reference) --------
    # Best-fit (for reference):
    # k1≈5.07e-3, k2≈2.10e-4, k3≈6.04e-5, k4≈7.99e-7
    # A0≈0.101, C0≈0.668, D0≈0.859, E0≈0.03035
    # Below we choose reasonably wide but focused ranges around these values.
    K1_LO, K1_HI = Number(1e-6), Number(5e-1)    # around 5e-3
    K2_LO, K2_HI = Number(1e-8), Number(1e-1)    # around 2e-4
    K3_LO, K3_HI = Number(1e-7), Number(1e-1)    # around 6e-5
    K4_LO, K4_HI = Number(1e-9), Number(1e-1)    # around 8e-7

    A0_LO, A0_HI = Number(1e-8), Number(1.0)     # 0.101 in range
    C0_LO, C0_HI = Number(0.0),  Number(0.0)    # 固定為 0（生成物基線）
    D0_LO, D0_HI = Number(1e-8), Number(1.5)     # 0.859 in range
    E0_LO, E0_HI = Number(0.0),  Number(0.0)    # 固定為 0（生成物基線）

    # -------- PDE (R1 + R2；R3 關閉) --------
    class Primary_Reaction_PDE(PDE):
        def __init__(self, Binit, dt_seconds):
            x_var = Symbol("x")  # <<<--- 修改點 1: 將 Symbol("x") 存為變數以便重用
            input_variables = {"x": x_var}

            def wrap(val):
                if isinstance(val, str):
                    return Function(val)(*input_variables)
                elif isinstance(val, (float, int)):
                    return Number(val)
                return val

            Binit_s = wrap(Binit)
            DTn = Number(dt_seconds)  # 使殘差單位一致（秒）

            # raw NN outputs
            raw_R  = Function("raw_R")(*input_variables)
            raw_R2 = Function("raw_R2")(*input_variables)

            # learnable initials (raw)
            A0_raw = Function("A0_raw")(*input_variables)
            C0_raw = Function("C0_raw")(*input_variables)
            D0_raw = Function("D0_raw")(*input_variables)
            E0_raw = Function("E0_raw")(*input_variables)

            sig = lambda z: Number(1)/(Number(1)+exp(-z))

            # bounded initials
            A0 = A0_LO + (A0_HI - A0_LO) * sig(A0_raw)
            # >>> 生成物基線固定 0
            C0 = Number(0.0)
            D0 = D0_LO + (D0_HI - D0_LO) * sig(D0_raw)
            # >>> 生成物基線固定 0
            E0 = Number(0.0)

            # <<<--- 修改點 2: 強制 R(0)=0 和 R2(0)=0
            # bounded progress variables with hard-enforced IC
            R  = x_var * (Number(1)/(Number(1)+exp(-raw_R ))) * Binit_s
            R2 = x_var * (Number(1)/(Number(1)+exp(-raw_R2))) * D0

            # measured B(t)
            B_meas = Function("B_meas")(*input_variables)

            # rate constants (bounded)
            k1 = K1_LO + (K1_HI - K1_LO) * sig(Function("k1")(*input_variables))
            k2 = K2_LO + (K2_HI - K2_LO) * sig(Function("k2")(*input_variables))
            k3 = K3_LO + (K3_HI - K3_LO) * sig(Function("k3")(*input_variables))
            k4 = K4_LO + (K4_HI - K4_LO) * sig(Function("k4")(*input_variables))

            # H definition with soft-positive and cap
            delta = Number(1e-10)
            H_raw = (Binit_s - R - Number(9)*R2)
            H_pos_uncapped = (H_raw + ((H_raw**2) + delta)**(Number(1)/2))/Number(2) + Number(EPS)
            H_pos = sympy.Min(H_pos_uncapped, Number(2.0))
            H_pow = (H_pos ** Number(9.0))

            # soft prior: k1>=k2, k3>=k4
            def pos(z): return (z + (z**2)**(Number(1)/2))/Number(2)
            order_norm = (pos(k2 - k1) + pos(k4 - k3)) / Number(1e-2)

            # weights (更直觀：數值越大→權重越重)
            weight_R, weight_R2, weight_stoich = Number(0.35), Number(0.32), Number(0.33)

            # unit-consistent residuals: d()/dx - DT * RHS
            reaction_R  = R.diff(x_var)  - DTn * (k1*(A0 - R)*H_pos              - k2*(C0 + R + Number(9)*R2))
            reaction_R2 = R2.diff(x_var) - DTn * (k3*(D0 - R2)*(H_pow)           - k4*((C0 + R + Number(9)*R2)**Number(9.0))*((E0 + Number(4)*R2)**Number(4.0)))
            stoich = H_raw - B_meas  

            self.equations = {
                "reaction_R_weighted":  reaction_R  * weight_R,
                "reaction_R2_weighted": reaction_R2 * weight_R2,
                "stoich_weighted":      stoich      * weight_stoich,
                "order_norm":           order_norm,
                "R":  R,
                "R2": R2,
            }

    # geometry & PDE
    geo = Line1D(0.0, 1.0)
    Primary_Reaction = Primary_Reaction_PDE(Binit=B_init, dt_seconds=DT)

    # -------- networks --------
    FC = instantiate_arch(
        cfg=cfg.arch.fully_connected,
        input_keys=[Key("x")],
        output_keys=[Key("raw_R"), Key("raw_R2")],
    )
    rate_constant1 = instantiate_arch(
        cfg=cfg.arch.linear, input_keys=[Key("c1")],
        output_keys=[Key("k1"), Key("k2")],
    )
    rate_constant2 = instantiate_arch(
        cfg=cfg.arch.linear, input_keys=[Key("c2")],
        output_keys=[Key("k3"), Key("k4")],
    )
    param_head = instantiate_arch(
        cfg=cfg.arch.linear, input_keys=[Key("cA")],
        output_keys=[Key("A0_raw"), Key("C0_raw"), Key("D0_raw"), Key("E0_raw")],
    )

    # init bias for stable start
    try:
        import torch
        m = find_torch_module(FC)
        if m is not None:
            core = get_core_module(m)
            last = None
            for mm in core.modules():
                if isinstance(mm, torch.nn.Linear): last = mm
            if last is not None and last.bias is not None and last.bias.numel() >= 2:
                with torch.no_grad():
                    last.bias.fill_(-1.0); last.bias[0] = -6.0  # R≈0, R2 small
        p("[INIT] last FC bias set (raw_R≈-6, raw_R2≈-1)")
    except Exception as e:
        p("[INIT] bias tweak skipped:", e)

    nodes = [
        FC.make_node(name="FC"),
        rate_constant1.make_node(name="rate_constant1"),
        rate_constant2.make_node(name="rate_constant2"),
        param_head.make_node(name="param_head"),
    ] + Primary_Reaction.make_nodes()

    # -------- Full Training --------
    domain = Domain()

    # <<<--- 修改點 3: 移除整個 IC (Initial Condition) 的 PointwiseBoundaryConstraint
    # 因為初始條件已經被硬性強制，不再需要這個軟性約束
    # IC: R(0)=0, R2(0)=0
    # IC = PointwiseBoundaryConstraint(
    #     nodes=nodes, geometry=geo,
    #     outvar={"R": 0.0, "R2": 0.0},
    #     batch_size=getattr(cfg.batch_size, "IC", 64),
    #     parameterization={Symbol("x"): 0.0, "c1": 1.0, "c2": 1.0, "cA": 1.0},
    # )
    # domain.add_constraint(IC, "IC")

    # PDE residual for R
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar={"x": x_aug, "c1": np.ones_like(x_aug), "c2": np.ones_like(x_aug), "cA": np.ones_like(x_aug)},
            outvar={"reaction_R_weighted": zeros},
            batch_size=getattr(cfg.batch_size, "residual_R", x_aug.shape[0]),
        ),
        "residual_R",
    )
    # PDE residual for R2 + prior
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar={"x": x_aug, "c1": np.ones_like(x_aug), "c2": np.ones_like(x_aug), "cA": np.ones_like(x_aug)},
            outvar={"reaction_R2_weighted": zeros, "order_norm": zeros},
            batch_size=getattr(cfg.batch_size, "residual_R2", x_aug.shape[0]),
        ),
        "residual_R2",
    )
    # Stoichiometry (data fit)
    domain.add_constraint(
        PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar={"x": x_aug, "B_meas": B_aug, "c1": np.ones_like(x_aug), "c2": np.ones_like(x_aug), "cA": np.ones_like(x_aug)},
            outvar={"stoich_weighted": zeros},
            batch_size=getattr(cfg.batch_size, "stoich", x_aug.shape[0]),
        ),
        "stoich",
    )

    total_steps = int(getattr(cfg.training, "max_steps", 200000))
    p(f"[TRAINING] Starting full model training for {total_steps} steps.")
    solver = Solver(cfg, domain)
    solver.solve()
    p("[DONE] training")

    # -------- POST: 取參數 / 正確的 ODE forward / 輸出 --------
    try:
        import torch, matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def sigmoid_np(z): return 1.0/(1.0+np.exp(-z))

        # ---- 取 k1..k4 ----
        m1 = get_core_module(find_torch_module(rate_constant1)); m1.eval()
        m2 = get_core_module(find_torch_module(rate_constant2)); m2.eval()
        with torch.no_grad():
            raw12 = m1(torch.tensor([[1.0]], dtype=torch.float32, device=module_device(m1))).cpu().numpy().ravel()
            raw34 = m2(torch.tensor([[1.0]], dtype=torch.float32, device=module_device(m2))).cpu().numpy().ravel()
        # 使用與 PDE 相同的上下界
        k1 = float(K1_LO) + (float(K1_HI) - float(K1_LO))*sigmoid_np(raw12[0])
        k2 = float(K2_LO) + (float(K2_HI) - float(K2_LO))*sigmoid_np(raw12[1])
        k3 = float(K3_LO) + (float(K3_HI) - float(K3_LO))*sigmoid_np(raw34[0])
        k4 = float(K4_LO) + (float(K4_HI) - float(K4_LO))*sigmoid_np(raw34[1])
        p(f"[POST] k1={k1:.6e}, k2={k2:.6e}, k3={k3:.6e}, k4={k4:.6e}")

        # ---- 取 A0..E0 ----
        mph = get_core_module(find_torch_module(param_head)); mph.eval()
        with torch.no_grad():
            raw = mph(torch.tensor([[1.0]], dtype=torch.float32, device=module_device(mph))).cpu().numpy().ravel()
        A0 = float(A0_LO) + (float(A0_HI) - float(A0_LO))*sigmoid_np(raw[0])
        # 與 PDE 一致：C0 固定為 0
        C0 = 0.0
        D0 = float(D0_LO) + (float(D0_HI) - float(D0_LO))*sigmoid_np(raw[2])
        # 與 PDE 一致：E0 固定為 0
        E0 = 0.0
        p(f"[POST] A0={A0:.6e}, C0={C0:.6e}, D0={D0:.6e}, E0={E0:.6e}")

        # ---- NN 前傳（R,R2,H）----
        mfc = get_core_module(find_torch_module(FC)); mfc.eval()
        with torch.no_grad():
            out = mfc(torch.tensor(x[:,None], dtype=torch.float32, device=module_device(mfc))).cpu().numpy()
        R_nn  = x * (1.0/(1.0+np.exp(-np.clip(out[:,0],-60,60))) * B_init)
        R2_nn = x * (1.0/(1.0+np.exp(-np.clip(out[:,1],-60,60))) * D0)
        H_nn  = np.maximum(B_init - R_nn - 9.0*R2_nn, 0.0) + EPS

        # ---- ODE forward (on REAL TIME t) ----
        def rk4_states_on_t(t_grid, R0=0.0, R20=0.0):
            R  = np.zeros_like(t_grid); R[0]  = R0
            R2 = np.zeros_like(t_grid); R2[0] = R20
            delta = 1e-10
            for i in range(len(t_grid)-1):
                dt = float(t_grid[i+1] - t_grid[i])
                def rhs(r, r2):
                    H_raw = B_init - r - 9.0*r2
                    H_pos = (H_raw + np.sqrt(H_raw*H_raw + delta))/2.0 + EPS
                    dR  = k1*(A0 - r)*H_pos  - k2*(C0 + r+9.0*r2)
                    dR2 = k3*(D0 - r2)*(H_pos**9.0) - k4*((C0 + r+9.0*r2)**9)*((E0 + 4.0*r2)**4.0)
                    return dR, dR2
                k1r,k1r2 = rhs(R[i], R2[i])
                k2r,k2r2 = rhs(R[i] + 0.5*dt*k1r,  R2[i] + 0.5*dt*k1r2)
                k3r,k3r2 = rhs(R[i] + 0.5*dt*k2r,  R2[i] + 0.5*dt*k2r2)
                k4r,k4r2 = rhs(R[i] + dt*k3r,      R2[i] + dt*k3r2)
                R[i+1]  = np.clip(R[i]  + (dt/6.0)*(k1r + 2*k2r + 2*k3r + k4r),  0.0, B_init)
                R2[i+1] = np.clip(R2[i] + (dt/6.0)*(k1r2 + 2*k2r2 + 2*k3r2 + k4r2), 0.0, max(D0, 1e-12))
            return R, R2

        # 使用位移後的時間軸 (t_shifted) 進行積分
        t_shifted = t - t0
        R_ode, R2_ode = rk4_states_on_t(t_shifted, 0.0, 0.0)
        H_ode  = np.maximum(B_init - R_ode - 9.0*R2_ode, 0.0) + EPS

        # ---- 輸出 ----
        artifacts = os.path.join(os.getcwd(), "artifacts"); os.makedirs(artifacts, exist_ok=True)
        out_df = pd.DataFrame({
            "time_sec": t, "t_norm": x,
            "H_meas": B_meas,
            "R_nn": R_nn, "R2_nn": R2_nn, "H_nn": H_nn,
            "R_ode": R_ode, "R2_ode": R2_ode, "H_ode": H_ode,
        })
        csv_out = os.path.join(artifacts, "predictions.csv")
        out_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
        p(f"[POST] saved -> {csv_out}")

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(dpi=150)
        ax.plot(t, B_meas, "o", ms=3, label="H_meas (data)")
        ax.plot(t, H_nn,  "-", lw=2, label="H_pred (NN)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Concentration (mol/L)")
        ax.set_title("H_pred (NN) vs H_meas"); ax.grid(True); ax.legend()
        fig.savefig(os.path.join(artifacts,"H_vs_time_NN.png"), bbox_inches="tight"); plt.close(fig)

        fig,ax = plt.subplots(dpi=150)
        ax.plot(t, B_meas, "o", ms=3, label="H_meas (data)")
        ax.plot(t, H_ode, "-", lw=2, label="H_pred (ODE forward; corrected)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Concentration (mol/L)")
        ax.set_title("H_pred (ODE forward with learned params) vs H_meas")
        ax.grid(True); ax.legend()
        fig.savefig(os.path.join(artifacts,"H_vs_time_ODE.png"), bbox_inches="tight"); plt.close(fig)

        err = np.abs(H_ode - B_meas)
        p(f"[CHECK][ODE] mean|H_ode-H_meas|={err.mean():.3e}, max={err.max():.3e}")

        # ---- 嘗試輸出 loss 變化圖 ----
        def try_plot_loss_curve():
            # 嘗試在 outputs/ 底下找常見命名的 CSV 檔，內含 'loss' 欄位
            patterns = [
                "outputs/**/*.csv",
                "outputs/**/rec_results*.csv",
                "outputs/**/results*.csv",
                "outputs/**/loss*.csv",
                "outputs/**/metrics*.csv",
                "outputs/**/train*.csv",
            ]
            candidates = []
            for pat in patterns:
                candidates.extend(glob.glob(pat, recursive=True))
            chosen = None
            for path in candidates:
                try:
                    dfc = pd.read_csv(path)
                    cols = [c.lower() for c in dfc.columns]
                    if any("loss" in c for c in cols):
                        chosen = (path, dfc); break
                except Exception:
                    continue
            if chosen is None:
                p("[LOSS] No suitable CSV with 'loss' column was found under outputs/. Skipped plotting.")
                return
            path, dfc = chosen
            # 嘗試找 step/epoch 欄；若無則用 index
            step_col = None
            for c in dfc.columns:
                cl = c.lower()
                if cl in ("step","global_step","steps","epoch","iter","iteration"):
                    step_col = c; break
            steps = dfc[step_col].to_numpy() if step_col is not None else np.arange(len(dfc))
            # 取第一個含 'loss' 的欄位當總損失
            loss_col = None
            for c in dfc.columns:
                if "loss" in c.lower():
                    loss_col = c; break
            if loss_col is None:
                p(f"[LOSS] CSV {path} has no 'loss' column after all. Skipped.")
                return
            fig, ax = plt.subplots(dpi=150)
            ax.plot(steps, dfc[loss_col].to_numpy(), "-", lw=1.5)
            ax.set_xlabel(step_col or "index")
            ax.set_ylabel(loss_col)
            ax.set_title("Training loss curve")
            ax.grid(True)
            fig.savefig(os.path.join(artifacts, "loss_curve.png"), bbox_inches="tight")
            plt.close(fig)
            p(f"[LOSS] Loss curve saved -> {os.path.join(artifacts, 'loss_curve.png')}")
        try_plot_loss_curve()

    except Exception as e:
        p("[POST] failed:", e)


if __name__ == "__main__":
    t_start = time.time()
    run()
    print(f"Total time: {time.time()-t_start:.2f}s")