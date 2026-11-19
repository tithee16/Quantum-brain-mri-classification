# QuantumMRI – hybrid quantum-classical brain-MRI classifier (Tkinter)

Desktop app that classifies brain MRI slices with a **variational quantum circuit (VQC)** followed by a
**classical softmax output layer**, and compares it against classical MLPs.

```
python main.py            # launch the GUI  (needs Tkinter – ships with python.org / most installers)
python train_cli.py --classes 4 --qubits 4 --layers 3 --epochs 40 --out models/default.npz   # headless training
```
Install: `pip install -r requirements.txt` (only NumPy, Pillow, Matplotlib; `tkinterdnd2` is optional and enables
true drag-and-drop – without it the drop zone is click-to-browse).

## What the app does
| Tab | Features |
|---|---|
| **Classify** | drag & drop / browse an image (or generate a synthetic sample) → preprocessing thumbnails (gray, histogram-equalised, Sobel edges, hot-spot map) → the PCA features shown as rotation angles → **animated quantum circuit** (gate-by-gate playhead, live ⟨Z⟩ read-out per qubit, hover a gate for its angle) → prediction ring with **confidence**, per-class probability bars for the quantum and the classical model |
| **Training** | dataset / circuit / optimiser settings, live loss + accuracy curves (train & validation) for all three models, progress bar, log; Stop button; runs in a background thread |
| **Compare** | table (params, train/test accuracy, precision, recall, F1, train time), grouped metric bars, parameter counts, confusion matrices, auto-written verdict, “Save charts…” |

## The model
1. **Pre-processing** – grayscale 64×64, percentile contrast stretch.
2. **Features** (50) – intensity statistics, left/right asymmetry, hot-spot area/position/spread, gradient stats, 4×4 intensity & hot-spot grids → standardise → **PCA to `n_qubits`** → min-max → angles in [0, π].
3. **Quantum feature map** – `ZZ` (H, RZ(x), RZZ((π−xᵢ)(π−xⱼ)/π), Qiskit-style) or plain `RY(x)` angle encoding.
4. **Variational ansatz** – L layers of RY, RZ on every qubit + CNOT ring (2·L·n trainable angles).
5. **Measurement** ⟨Z⟩ on each qubit → **classical dense + softmax** layer.
6. **Training** – Adam; circuit gradients by the exact **parameter-shift rule** (checked against finite differences: error ~1e-11).
7. **Baselines** – NumPy MLP with *matched* parameter count (fair comparison) and an MLP on all 50 features.

## Honest notes
* **Engine:** this build uses its own exact NumPy state-vector simulator instead of PennyLane/Qiskit + PyTorch, so the
  app has no heavy dependencies and runs anywhere. The circuit description (`qmri/quantum.py: build_ops`) is plain data
  and maps one-to-one onto a PennyLane `QNode` / Qiskit `QuantumCircuit` if you want to port it or run on hardware.
* **Data:** the bundled `models/default.npz` is trained on **synthetic** MRI-like images (clearly not real anatomy).
  For real data use *Training → Image folder* with one sub-folder per class (the Kaggle “Brain Tumor MRI Dataset”
  layout with `Training/` and `Testing/` is detected automatically), then press *Train all models*.
* On the demo data the full-feature MLP wins; the quantum model is roughly on par with the parameter-matched MLP.
  No quantum advantage is claimed – the point is a transparent, inspectable hybrid pipeline.
* Simulation cost grows ~2ⁿ with qubits (≤6 is comfortable on a laptop).
* Research / education only – **not a medical device**.

## Layout
`qmri/quantum.py` simulator + hybrid model · `qmri/features.py` preprocessing/PCA · `qmri/classical.py` MLP ·
`qmri/data.py` synthetic + folder datasets · `qmri/trainer.py` training/metrics/save-load · `qmri/gui.py` Tkinter UI
