# fastMD

**Keep using ASE's `Atoms`, optimizers, and molecular dynamics. Just change the calculator.**

fastMD provides a common interface to CUDA Graph inference backends for MatRIS,
CHGNet, and ALIGNN. On a GPU, it attempts to capture and replay energy and force
calculations by default. On a CPU, or for unsupported properties, it uses eager
inference and reports the reason. Capacity buckets, warmup, and kernel fusion
have defaults, so everyday use requires no CUDA Graph tuning.

```python
from ase.build import bulk
from fastmd import FastMDCalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = FastMDCalculator("chgnet")

print(atoms.get_potential_energy())  # Total energy, eV
print(atoms.get_forces())            # (N, 3), eV/Å
```

This package is independent of the adjacent `MatRIS-09bk` directory. Installation
and use do not require switching old branches, changing `PYTHONPATH`, or
installing three overlapping source repositories. CHGNet 0.3.0 weights are
bundled, so the example above can run offline.

## 1. Installation

Python 3.11 or newer is required. Install an appropriate PyTorch build in your
environment first, then install the dependencies for your model:

```bash
cd fastMD
python -m pip install -e '.[chgnet]'
# Alternatively, install dependencies for other models
python -m pip install -e '.[matris]'
python -m pip install -e '.[alignn]'
```

CUDA Graph requires an NVIDIA GPU, CUDA-enabled PyTorch 2.8 or newer, a matching
Triton version, and the GPU neighbor-list operators:

```bash
python -m pip install -e '.[matris,chgnet,cuda]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The `cuda` extra does not guarantee that an existing CPU-only PyTorch installation
will be replaced with the appropriate CUDA build. Install
[PyTorch](https://pytorch.org/get-started/locally/) for your machine first.
CUDA Graph itself is provided by PyTorch. CPU inference does not require Warp,
Triton, or NVIDIA operators, and does not require compiling the old Cython/CUDA
extensions.

ALIGNN also requires DGL. The declared dependency uses the DGL 1.x interface; CUDA
execution requires a DGL build compatible with your PyTorch/CUDA environment.
Validate these dependencies in a separate ALIGNN environment before attempting
to combine existing model environments.

## 2. Models and checkpoints

All models use `FastMDCalculator(model, checkpoint=..., device=...)`.

| Model | Default or local checkpoint | Eager properties | CUDA Graph properties |
| --- | --- | --- | --- |
| `matris` | Downloads `matris_10m_oam` by default; or a local `.pth.tar` file | Energy, forces, stress, magnetic moments | Energy, forces, stress, magnetic moments |
| `chgnet` | Bundled 0.3.0 weights; or a local `.pth.tar` file | Energy, forces, stress, magnetic moments | Energy, forces |
| `alignn` | Requires a directory containing `config.json` and `best_model.pt` | Energy, forces | Energy, forces |
| `dpa4` | Requires the released DPA4 checkpoint and WBM runtime | Energy, forces | Energy, forces |
| `nequip` | Requires the released NequIP checkpoint and WBM runtime | Energy, forces | Energy, forces |
| `orbv3` | Requires the released ORB-v3 checkpoint and WBM runtime | Energy, forces | Energy, forces |
| `sevennet` | Requires the released SevenNet checkpoint and WBM runtime | Energy, forces | Energy, forces |
| `tace` | Requires the released TACE checkpoint and WBM runtime | Energy, forces | — |

This table describes the implemented interfaces. See the
[validation record](docs/validation.md) for what has been tested on the current
node. The CUDA paths were migrated from the original branches and still need
the numerical consistency checks below on your target GPU.

The five WBM adapters are documented separately in
[WBM backends](docs/wbm_backends.md). Their model packages remain optional and
are imported only when the corresponding model is selected.

```python
from fastmd import FastMDCalculator

# Local MatRIS checkpoint, fully offline
calc = FastMDCalculator("matris", checkpoint="../checkpoint/MatRIS_10M_OAM.pth.tar")

# CHGNet 0.3.0 is bundled by default
calc = FastMDCalculator("chgnet")
# Your own trained CHGNet checkpoint
calc = FastMDCalculator("chgnet", checkpoint="checkpoints/my_chgnet.pth.tar")

# ALIGNN expects a directory, not a single .pt file
calc = FastMDCalculator("alignn", checkpoint="checkpoints/v12.2.2024_dft_3d_307k")
```

MatRIS retains the original download mechanism, with a cache at `~/.cache/matris`.
Use `model_kwargs={"model_name": "matris_10m_mp"}` to select the other supported
default model. For other CHGNet versions, provide `checkpoint`; only 0.3.0 is
bundled.

The ALIGNN adapter currently supports `alignn_atomwise` potentials. Capture
requires the `radius_graph` neighbor strategy, at least one ALIGNN layer, and a
supported model configuration. Unsupported configurations explicitly fall back
to eager inference in `auto` mode.

ALIGNN retains the original branch's total-energy and force-scaling conventions.
For other training configurations, use `model_kwargs` to set `intensive`,
`force_multiplier`, `force_mult_natoms`, and `force_mult_batchsize`, and check the
scaling against your reference forces. Their defaults are `True`, `1.0`, `False`,
and `True`, respectively.

The local `v12.2.2024_dft_3d_307k` configuration has `batch_size=6`. With the
original interface's defaults, the returned forces are six times the unscaled
negative energy gradient. fastMD retains this behavior for compatibility with
the original branch. To obtain forces consistent with the returned energy's
negative gradient, explicitly set
`model_kwargs={"force_mult_batchsize": False}` and verify the checkpoint's
training and inference conventions. In particular, do not assume that the
original default scaling guarantees energy conservation in NVE simulations.

## 3. Using standard ASE workflows

### Single-point calculations

```python
from ase.io import read
from fastmd import FastMDCalculator

atoms = read("POSCAR")
atoms.calc = FastMDCalculator("chgnet", device="auto")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
# CHGNet stress and magnetic moments use eager inference in default auto mode
stress = atoms.get_stress()                 # xx, yy, zz, yz, xz, xy; eV/Å³
magmoms = atoms.get_magnetic_moments()       # μB
print(atoms.calc.stats())                   # Actual mode, fallback reason, cache statistics
```

All backends return total energy; you do not need to multiply it by the atom
count. `free_energy` equals `energy`, supporting ASE optimizers that request
`force_consistent=True`. No additional electronic-temperature free-energy
correction is applied.

The three adapters currently target fully periodic materials with a nonzero
cell volume (`pbc=True`). Nonperiodic molecules and partially periodic systems
raise an explicit error. The adapters do not silently alter your vacuum spacing,
cell, or PBC settings.

### Molecular dynamics

```python
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from fastmd import FastMDCalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms.calc = FastMDCalculator("chgnet")
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))

# Optional: warm up and capture the current geometry before timing. Atoms do not move.
atoms.calc.warmup(atoms)

dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300,
               friction=0.01 / units.fs, trajectory="md.traj",
               logfile="md.log", loginterval=10)
dyn.run(1000)
print(atoms.calc.stats())
```

Use standard ASE units: multiply a time step in femtoseconds by `units.fs`, and
pass temperature in kelvin through `temperature_K`. ASE constraints,
`dyn.attach()`, trajectory output, and NVE/NVT workflows work as usual.

The acceleration here applies to **model inference**. ASE still runs the
integrator and Python loop, and each call still involves host/device data
transfers. This release does not expose the original branches' whole-step GPU MD
as ASE MD or capture the entire MD loop in a CUDA Graph. Initial capture has a
setup cost, so short jobs may not benefit. Benchmark your own system.

### Geometry and cell relaxation

```python
from ase.optimize import FIRE

# Fixed cell: supported by all three backends
FIRE(atoms, trajectory="relax.traj").run(fmax=0.05, steps=500)

# Variable cell: use MatRIS or CHGNet, which provide stress
from ase.filters import FrechetCellFilter
FIRE(FrechetCellFilter(atoms)).run(fmax=0.05, steps=500)
```

MatRIS can capture stress calculations. CHGNet computes stress eagerly; use
`cuda_graph=False` for cell relaxation to avoid unnecessary capture overhead.
ALIGNN currently does not expose stress through this interface and cannot be
used for NPT or cell relaxation here. Cell changes invalidate captures, so
variable-cell workflows can trigger frequent recapture and may perform better
with eager inference.

## 4. CUDA Graph defaults

Most users can keep the defaults:

```python
calc = FastMDCalculator("chgnet")                    # auto, the default
calc = FastMDCalculator("chgnet", cuda_graph=False)  # Eager inference for comparison
calc = FastMDCalculator("chgnet", cuda_graph=True)   # Require capture; raise if unavailable
calc = FastMDCalculator("chgnet", device="cpu")      # Explicit CPU execution
calc = FastMDCalculator("chgnet", device="cuda:1")   # Select a GPU
```

`auto` falls back only for known capability limitations or missing dependencies
and reports the reason. Unexpected runtime errors, out-of-memory errors, and
invalid model outputs propagate to the caller. With `True`, requesting stress
or magnetic moments that the backend cannot capture also raises an error.

For manual tuning:

```python
from fastmd import FastMDCalculator, CUDAGraphConfig

calc = FastMDCalculator("chgnet", cuda_graph=CUDAGraphConfig(
    enabled="auto",
    warmup_steps=3,
    max_cached_graphs=8,
    edge_capacity_step=None,       # None = model-specific default
    triplet_capacity_step=None,
    enable_fusions=True,
))
```

| Setting | MatRIS default | CHGNet default | ALIGNN default |
| --- | --- | --- | --- |
| Warmup iterations before capture | 3 | 3 | 3 |
| Edge capacity increment | 512 undirected edges | 128 undirected edges | 1024 directed edges |
| Triplet capacity increment | 8192 | 1024 | 16384 |
| Cache policy | Multiple buckets; clear after exceeding 8 | Multiple buckets; clear after exceeding 8 | Keep one capacity; recapture on overflow |
| Kernel fusion | Enabled | Enabled | Enabled |

Larger capacity increments can reduce recapture frequency but increase memory
use and padded computation. When MatRIS or CHGNet exceeds the cache limit, it
finishes the current evaluation, clears captures, and captures again on the next
call. This limit is not a hard GPU memory quota.

These public settings apply to each calculator instance. You do not need to set
`MATRIS_*` or `CHGNET_*` environment variables. The migrated kernels retain a few
internal experimental environment variables, which ordinary workflows should
not depend on.

Changes to atom count, species, species order, cell, or PBC automatically
invalidate existing captures. Neighbor lists are rebuilt for the new positions
on each evaluation. When edge or triplet counts exceed capacity, the backend
increases capacity and captures again. Arrays returned to ASE own their storage
and are not overwritten by the next replay.

```python
print(calc.stats())     # mode: not-run / eager / cuda_graph
calc.clear_cache()      # Release captures and ASE results; keep model weights
```

Use one calculator/backend per serial workflow; do not share an instance across
threads. `warmup()` prepares the cache for the current geometry only. Subsequent
changes in temperature or neighbor counts can still trigger new captures.

## 5. Examples, numerical checks, and timing

After installation, run these commands from this directory:

```bash
python examples/single_point.py --model chgnet
python examples/md.py --model chgnet --steps 100
python examples/compare.py --model chgnet
python examples/compare.py --model matris --checkpoint ../checkpoint/MatRIS_10M_OAM.pth.tar
```

`compare.py` runs eager and CUDA Graph inference on the same perturbed geometries
on a GPU, checks energy and force differences, and reports warmup and subsequent
evaluation times separately. It exits explicitly when no GPU is available;
CPU execution is not reported as a CUDA acceleration result.

```bash
python -m pip install -e '.[chgnet,test]'
python -m pytest -q
# Include tests for local MatRIS and ALIGNN checkpoints
FASTMD_MATRIS_CHECKPOINT=/absolute/path/model.pth.tar \
FASTMD_ALIGNN_CHECKPOINT=/absolute/path/alignn_directory \
python -m pytest -q
```

CUDA tests are skipped when no GPU is available. Model integration tests are
skipped when their local checkpoints are missing. CUDA tests cover replay,
position perturbations, cell changes, composition changes, atom-count changes,
and the lifetime of returned arrays.

Assess performance and numerical tolerances for your model, system, GPU, and
trajectory length. Long MD trajectories are not expected to remain bitwise
identical across execution modes.

## 6. Adding more models for WBM workflows

The [model integration guide](docs/adding_models.md) provides a runnable
registration example and an integration checklist. Users keep the same API:
`FastMDCalculator("new_model", checkpoint=...)`. Developers put model loading,
graph representations, and capture details in `models/<name>.py`; the ASE
calculator does not need to change.

The architecture draws on the common capability declarations and adapter layer
in [torch-sim's model interface](https://github.com/TorchSim/torch-sim/blob/main/torch_sim/models/interface.py).
It does not depend on torch-sim or implement its batching interface. The current
API handles one ASE `Atoms` object at a time. For WBM screening, you can initially
reuse a calculator across structures, with captures rebuilt according to the
invalidation rules. True batching and capacity scheduling across structures
remain separate future work.

Performance on WBM does not establish that a model provides forces, stress, or
safe CUDA capture. Declare each model's capabilities explicitly and validate
total energy, forces, stress sign, and units against its original calculator.

## 7. Repository layout and provenance

```text
fastMD/
├── src/fastmd/
│   ├── calculator.py      # Common ASE entry point
│   ├── config.py          # Defaults for each calculator instance
│   ├── models/            # Backend contract, registry, and three adapters
│   └── _vendor/           # Migrated models, graph builders, and optimized kernels
├── examples/              # Single-point, ASE MD, and CUDA accuracy/timing comparisons
├── tests/                 # ASE contract and real-model integration tests
├── docs/                  # Architecture, integration guide, provenance, and validation
└── licenses/              # Original project licenses
```

The [migration notes](docs/migration.md) describe the original branches'
responsibilities and the fixes made during this refactor. The
[source manifest](docs/sources.json) records each branch's commit and migrated
files. Private namespaces avoid conflicts with `matris`, `chgnet`, and `alignn`
packages already installed in your environment. Original license and citation
requirements continue to apply; see [NOTICE](NOTICE) and `licenses/`.
