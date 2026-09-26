"""Real checkpoints: CPU tests plus opt-in CUDA replay regression tests."""
import os

import numpy as np
import pytest
import torch
from ase.build import bulk

from fastmd import FastMDCalculator


def checkpoint_for(name):
    if name == "chgnet":
        pytest.importorskip("pymatgen")
        return None
    checkpoint = os.environ.get(f"FASTMD_{name.upper()}_CHECKPOINT")
    if not checkpoint:
        pytest.skip(f"Set FASTMD_{name.upper()}_CHECKPOINT to test a local model")
    return checkpoint


@pytest.mark.integration
@pytest.mark.parametrize("name", ["chgnet", "matris", "alignn"])
def test_real_model_cpu(name):
    torch.set_num_threads(2)
    calc = FastMDCalculator(name, checkpoint=checkpoint_for(name), device="cpu", cuda_graph=False)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions[0] += [.02, -.01, .015]
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (8, 3)
    # Check force sign and extensive energy against a finite difference.
    h = 2e-3
    atoms.positions[0, 0] += h
    plus = atoms.get_potential_energy()
    atoms.positions[0, 0] -= 2*h
    minus = atoms.get_potential_energy()
    # ALIGNN's upstream calculator multiplies forces by training batch_size.
    # Check the unscaled model gradient without silently changing that API.
    force_scale = calc.backend.calculator.config["batch_size"] if name == "alignn" else 1
    np.testing.assert_allclose(forces[0, 0] / force_scale, -(plus-minus)/(2*h), atol=.01, rtol=.05)
    assert calc.stats()["mode"] == "eager"


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("name", ["chgnet", "matris", "alignn"])
def test_cuda_replay_matches_eager(name):
    checkpoint = checkpoint_for(name)
    eager = FastMDCalculator(name, checkpoint=checkpoint, device="cuda", cuda_graph=False)
    graph = FastMDCalculator(name, checkpoint=checkpoint, device="cuda", cuda_graph=True)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    rng = np.random.default_rng(42)
    reference_forces = None
    for step in range(7):
        atoms.positions += rng.normal(0, .015, atoms.positions.shape)
        if step == 3:
            atoms.set_cell(atoms.cell * 1.03, scale_atoms=True)
        if step == 4:
            atoms.numbers[0] = 6  # composition-dependent captures must be invalidated
        if step == 5:
            atoms = atoms.repeat((2, 1, 1))
        atoms.calc = eager
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = graph
        np.testing.assert_allclose(atoms.get_potential_energy(), energy, atol=3e-3, rtol=1e-5)
        result = atoms.get_forces()
        np.testing.assert_allclose(result, forces, atol=3e-3, rtol=1e-3)
        if step == 0:
            reference_forces = result
            saved = result.copy()
    np.testing.assert_array_equal(reference_forces, saved)
    assert graph.stats()["mode"] == "cuda_graph"
    assert graph.stats()["graph_calls"] == 7
    assert graph.stats()["invalidations"] == 3


@pytest.mark.integration
def test_alignn_matches_original_calculator():
    checkpoint = checkpoint_for("alignn")
    from fastmd._vendor.alignn.ff.calculators import AlignnAtomwiseCalculator
    original = AlignnAtomwiseCalculator(path=checkpoint, device="cpu", graph_device="cpu", include_stress=False)
    calc = FastMDCalculator("alignn", checkpoint=checkpoint, device="cpu", cuda_graph=False)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions[0] += [.02, -.01, .015]
    original.calculate(atoms.copy())
    calc.calculate(atoms)
    np.testing.assert_allclose(calc.results["energy"], np.asarray(original.results["energy"]).item(), atol=1e-5)
    np.testing.assert_allclose(calc.results["forces"], original.results["forces"], atol=1e-5)


@pytest.mark.integration
def test_alignn_static_mask_matches_dynamic_on_cpu():
    """Exercise the actual captured numerical function even without a GPU."""
    checkpoint = checkpoint_for("alignn")
    from fastmd._vendor.alignn.ff.calculators import ase_to_atoms
    from fastmd._vendor.alignn.graphs import Graph
    from fastmd._vendor.alignn.ff.cuda_graph import static_masked_alignn_forward
    calc = FastMDCalculator("alignn", checkpoint=checkpoint, device="cpu", cuda_graph=False)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.positions[0] += [.03, -.02, .01]
    common = dict(atoms=ase_to_atoms(atoms), neighbor_strategy="radius_graph",
                  atom_features="atomic_number", use_canonize=True)
    graph, line_graph = Graph.atom_dgl_multigraph(**common, cutoff=4.0)
    candidate, candidate_line = Graph.atom_dgl_multigraph(**common, cutoff=4.5)
    assert candidate.num_edges() > graph.num_edges()
    cell = torch.as_tensor(atoms.cell.array, dtype=torch.float32)
    expected = calc.backend.model((graph, line_graph, cell))
    source, destination = candidate.edges()
    line_source, line_destination = candidate_line.edges()
    mask = torch.linalg.vector_norm(candidate.edata["r"], dim=1) < 4.0 + 1e-6
    forces, energy = static_masked_alignn_forward(
        calc.backend.model, node_features=candidate.ndata["atom_features"],
        source=source, destination=destination, line_source=line_source,
        line_destination=line_destination, displacement=candidate.edata["r"],
        edge_mask=mask[:, None].float(), triplet_mask=(mask[line_source] & mask[line_destination])[:, None].float(),
        operator_fusions=False,
    )
    torch.testing.assert_close(energy, expected["out"], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(forces, expected["grad"], atol=3e-5, rtol=1e-4)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["chgnet", "matris"])
def test_real_model_magmoms_and_stress(name):
    calc = FastMDCalculator(name, checkpoint=checkpoint_for(name), device="cpu", cuda_graph=False)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = calc
    assert atoms.get_magnetic_moments().shape == (2,)
    assert atoms.get_stress().shape == (6,)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["chgnet", "matris"])
def test_sink_padding_preserves_predictions_on_cpu(name):
    """Validate the padding used by CUDA Graph against a real ragged model."""
    from pymatgen.io.ase import AseAtomsAdaptor
    calc = FastMDCalculator(name, checkpoint=checkpoint_for(name), device="cpu", cuda_graph=False)
    model = calc.backend.model
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[0] += [.02, -.01, .015]
    graph = model.graph_converter(AseAtomsAdaptor.get_structure(atoms))
    if name == "chgnet":
        from fastmd._vendor.chgnet.model.cuda_graph import pad_crystal_graph
        padded, n_real = pad_crystal_graph(graph, graph.undirected2directed.shape[0] + 64,
                                           graph.bond_graph.shape[0] + 64)
        composition = model.composition_model([graph]).detach()
        expected = model([graph], task="ef")
        actual = model([padded], task="ef", n_real=n_real, composition_energy=composition)
    else:
        from fastmd._vendor.matris.applications.cuda_graph import pad_radius_graph
        padded, n_real = pad_radius_graph(graph, graph.undirected2directed.shape[0] + 64,
                                          graph.line_graph.shape[0] + 64)
        expected = model([graph], task="efs", is_training=False)
        actual = model([padded], task="efs", is_training=False, n_real=n_real)
        torch.testing.assert_close(actual["s"][0], expected["s"][0], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual["e"], expected["e"], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(actual["f"][0][:n_real], expected["f"][0], atol=1e-4, rtol=1e-3)
