# Modal (personal workspace `raihanulhaque007`)

Volume: `facevitals-data`, mounted at `/data` in every function.

```bash
conda run -n ML modal profile activate raihanulhaque007
```
```bash
conda run -n ML modal run modal_app/setup_data.py
```
Inspect / upload / download:
```bash
conda run -n ML modal volume ls facevitals-data /
```
```bash
conda run -n ML modal volume put facevitals-data data/raw/UBFC-rPPG /UBFC-rPPG
```
```bash
conda run -n ML modal volume get facevitals-data /rppg_biases/Ground_Truth data/raw/rppg_biases/
```
Request-only datasets (UBFC-rPPG, PURE, MMPD, COHFACE) go in with `modal volume put` after approval.
