"""Asynchronous ASE output for GPU-resident molecular dynamics."""

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
from torch import Tensor


@dataclass
class _FrameSlot:
    positions: Tensor
    momenta: Tensor
    forces: Tensor
    cell: Tensor
    scalars: Tensor
    event: torch.cuda.Event | None = None


@dataclass
class _PendingFrame:
    slot: _FrameSlot
    step: int
    time_ps: float


class AsyncGPUMDLogger:
    """Copy frames asynchronously and write ASE trajectory/log output."""

    def __init__(
        self,
        atoms_template: Atoms,
        device: torch.device,
        trajectory=None,
        logfile: str | Path | IO[str] | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        ring: int = 8,
    ) -> None:
        self.enabled = trajectory is not None or logfile is not None
        self.loginterval = max(int(loginterval), 1)
        self.device = device
        self.atoms_template = atoms_template.copy()
        self._stop = object()
        self._available: queue.Queue[_FrameSlot] = queue.Queue(maxsize=ring)
        self._pending: queue.Queue[_PendingFrame | object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._trajectory = None
        self._log = None
        self._close_log = False
        self._stream = None

        if not self.enabled:
            return
        if trajectory is not None:
            if isinstance(trajectory, (str, Path)):
                mode = "a" if append_trajectory else "w"
                self._trajectory = Trajectory(
                    str(trajectory),
                    mode,
                    self.atoms_template,
                )
            else:
                self._trajectory = trajectory
        if logfile is not None:
            if logfile == "-":
                self._log = sys.stdout
            elif isinstance(logfile, (str, Path)):
                self._log = open(logfile, "a", encoding="utf-8")
                self._close_log = True
            else:
                self._log = logfile
            self._write_log_header()

        pin_memory = device.type == "cuda"
        n_atoms = len(self.atoms_template)
        for _ in range(max(int(ring), 1)):
            self._available.put(
                _FrameSlot(
                    positions=torch.empty(
                        (n_atoms, 3),
                        dtype=torch.float32,
                        pin_memory=pin_memory,
                    ),
                    momenta=torch.empty(
                        (n_atoms, 3),
                        dtype=torch.float32,
                        pin_memory=pin_memory,
                    ),
                    forces=torch.empty(
                        (n_atoms, 3),
                        dtype=torch.float32,
                        pin_memory=pin_memory,
                    ),
                    cell=torch.empty(
                        (3, 3),
                        dtype=torch.float32,
                        pin_memory=pin_memory,
                    ),
                    scalars=torch.empty(
                        3,
                        dtype=torch.float64,
                        pin_memory=pin_memory,
                    ),
                )
            )
        if device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _write_log_header(self) -> None:
        if self._log is None:
            return
        self._log.write(
            "%-9s %12s %12s %12s  %6s\n"
            % ("Time[ps]", "Etot[eV]", "Epot[eV]", "Ekin[eV]", "T[K]")
        )
        self._log.flush()

    def should_log(self, step: int) -> bool:
        """Return whether output is requested for this MD step."""
        return self.enabled and step % self.loginterval == 0

    def enqueue(
        self,
        *,
        step: int,
        time_ps: float,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor,
        cell: Tensor,
        potential_energy: Tensor,
        kinetic_energy: Tensor,
        temperature: Tensor,
    ) -> None:
        """Enqueue nonblocking device-to-host copies for one frame."""
        if not self.enabled:
            return
        slot = self._available.get()
        scalars = torch.stack(
            (
                potential_energy.reshape(()).to(torch.float64),
                kinetic_energy.reshape(()).to(torch.float64),
                temperature.reshape(()).to(torch.float64),
            )
        )
        assert self._stream is not None
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._stream):
            slot.positions.copy_(positions.detach(), non_blocking=True)
            slot.momenta.copy_(momenta.detach(), non_blocking=True)
            slot.forces.copy_(forces.detach(), non_blocking=True)
            slot.cell.copy_(cell.detach(), non_blocking=True)
            slot.scalars.copy_(scalars.detach(), non_blocking=True)
            slot.event = torch.cuda.Event()
            slot.event.record(self._stream)
        self._pending.put(
            _PendingFrame(slot=slot, step=int(step), time_ps=float(time_ps))
        )

    def _writer_loop(self) -> None:
        while True:
            item = self._pending.get()
            if item is self._stop:
                self._pending.task_done()
                return
            assert isinstance(item, _PendingFrame)
            if item.slot.event is not None:
                item.slot.event.synchronize()
            self._write_frame(item)
            self._available.put(item.slot)
            self._pending.task_done()

    def _write_frame(self, frame: _PendingFrame) -> None:
        scalars = frame.slot.scalars.numpy()
        potential = float(scalars[0])
        kinetic = float(scalars[1])
        temperature = float(scalars[2])
        if self._trajectory is not None:
            atoms = self.atoms_template.copy()
            atoms.set_cell(frame.slot.cell.numpy(), scale_atoms=False)
            atoms.set_positions(
                np.array(frame.slot.positions.numpy(), copy=True)
            )
            atoms.set_momenta(np.array(frame.slot.momenta.numpy(), copy=True))
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=potential,
                forces=np.array(frame.slot.forces.numpy(), copy=True),
            )
            self._trajectory.write(atoms)
        if self._log is not None:
            self._log.write(
                "%10.4f %12.3f %12.3f %12.3f %6.1f\n"
                % (
                    frame.time_ps,
                    potential + kinetic,
                    potential,
                    kinetic,
                    temperature,
                )
            )
            self._log.flush()

    def flush(self) -> None:
        """Wait for all queued output to finish."""
        if self.enabled:
            self._pending.join()

    def close(self) -> None:
        """Flush and close output resources."""
        if not self.enabled:
            return
        self.flush()
        self._pending.put(self._stop)
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._trajectory is not None and hasattr(
            self._trajectory,
            "close",
        ):
            self._trajectory.close()
        if self._log is not None and self._close_log:
            self._log.close()
        self.enabled = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
