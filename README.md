# PINNs (PhysicsNeMo Inverse Fitting for Mg–Al–H Kinetics)

用 **PhysicsNeMo (Modulus-Sym)** 做 PINN（Physics-Informed Neural Networks）反推（inverse problem）反應動力學參數的最小可重現專案。

目前版本聚焦在 **Primary Reaction 的雙反應進度變數**：
- 反應進度：`R`（R1）與 `R2`（R3 關閉）
- 反推參數：`k1..k4`（以 sigmoid 做 bounded transform）
- 初始參數：`A0..E0`（其中 `C0 = 0`, `E0 = 0` 固定）
- IC（初始條件）採 **hard-enforced**：`R(0)=0`, `R2(0)=0`
- 損失設計改為 **加權（weight）** 而非 scale-normalization，並包含速率常數排序懲罰（order constraint）

---

## Repo Structure


---

## Model Summary (High-level)

此專案用 PINN 同時滿足：
1. **反應動力學 ODE/PDE 殘差**（R 與 R2 的微分方程）
2. **化學計量 / 觀測一致性**（例如 `stoich = H_raw - B_meas` 這類關係）
3. **參數範圍約束**：`k1..k4` 透過 sigmoid 映射到指定上下界
4. **參數結構偏好**：order penalty（偏好 `k1 ≥ k2`、`k3 ≥ k4`）

> 重要：`C0` 與 `E0` 在目前版固定為 0（生成物基線）。若要開放為可訓練參數，需修改程式中 `*_LO`/`*_HI` 與 PDE 定義。

---

## Requirements

- Python 3.9+（建議 3.10）
- PyTorch（版本依你的 CUDA/環境而定）
- NVIDIA **PhysicsNeMo / Modulus-Sym**（或相容的 `physicsnemo.sym` 套件環境）
- 常用科學套件：numpy, pandas, sympy, matplotlib

> ⚠️ PhysicsNeMo 的安裝方式會依版本而不同（pip / conda / docker）。建議你用你平常跑 Modulus/PhysicsNeMo 的同一套環境。

---

## Installation (Minimal)

### Option A: 直接用你現有的 PhysicsNeMo 環境
確認能 import：

```bash
python -c "import physicsnemo.sym; print('OK')"

