"""MatRIS adapter using the migrated main-branch inference engine."""
from ase import units

from .base import ModelBackend, ModelCapabilities


class MatRISModel(ModelBackend):
    capabilities = ModelCapabilities(
        frozenset({"energy", "forces", "stress", "magmoms"}),
        frozenset({"energy", "forces", "stress", "magmoms"}),
    )

    def __init__(self, *, checkpoint=None, model_name="matris_10m_oam", **kwargs):
        super().__init__(**kwargs)
        from fastmd._vendor.matris.applications.base import MatRISCalculator
        self.calculator = MatRISCalculator(model_path=str(checkpoint) if checkpoint is not None else None,
                                          model=model_name, device=str(self.device), task="ef")
        self.model = self.calculator.model.eval()
        self.model.enable_checkpoint = False
        for layer in self.model.interaction_block:
            layer.enable_checkpoint = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # Honor explicit CPU even on hosts with CUDA and GPU neighbor operators.
        if self.device.type == "cpu":
            self.model.graph_converter.algorithm = "legacy"
            self.calculator._gpu_graph = False
        self.runner = None
        self._task = None

    @staticmethod
    def _prediction_task(properties):
        return "efsm" if "magmoms" in properties else ("efs" if "stress" in properties else "ef")

    def graph_unavailable_reason(self, properties):
        reason = super().graph_unavailable_reason(properties)
        if reason:
            return reason
        if self.model.reference_energy is None:
            return "MatRIS CUDA Graph requires a reference-energy table for isolated-atom handling"
        from fastmd._vendor.matris.graph.gpu_graph_builder import op_available
        return None if op_available else "GPU neighbor operators missing; install fastMD[cuda]"

    def _predict_eager(self, atoms, properties):
        calc = self.calculator
        calc.task = self._prediction_task(properties)
        calc.key = set(calc.task) | {"atoms_per_graph", "ref_energy"}
        calc.calculate(atoms.copy(), list(properties), ["positions"])
        return {k: v for k, v in calc.results.items() if v is not None and k in self.capabilities.properties}

    def _predict_graph(self, atoms, properties):
        from fastmd._vendor.matris.applications.cuda_graph import BucketedGraphRunner
        from fastmd._vendor.matris.graph.gpu_graph_builder import atoms_to_graph_gpu
        task = self._prediction_task(properties)
        if self.runner is None or task != self._task:
            self.runner = BucketedGraphRunner(
                self.model, task=task,
                u_step=self.config.edge_capacity_step or 512,
                t_step=self.config.triplet_capacity_step or 8192,
                warmup=self.config.warmup_steps,
                enable_model_fusions=self.config.enable_fusions,
            )
            self._task = task
        converter = self.model.graph_converter
        graph = atoms_to_graph_gpu(atoms, atom_graph_cutoff=converter.atom_graph_cutoff,
                                   line_graph_cutoff=converter.line_graph_cutoff, device=self.device)
        output, n = self.runner.run(graph)
        scale = n if self.model.is_intensive else 1
        results = {"energy": float(output["e"][0].detach()) * scale,
                   "forces": output["f"][0][:n].detach().cpu().numpy().copy()}
        if "stress" in properties:
            results["stress"] = output["s"][0].detach().cpu().numpy().copy() * units.GPa
        if "magmoms" in properties:
            results["magmoms"] = output["m"][0][:n].detach().cpu().numpy().copy()
        if len(self.runner.cache) > self.config.max_cached_graphs:
            self.clear_graphs()
        return results

    def clear_graphs(self):
        self.runner = None
        self._task = None

    def stats(self):
        return {**super().stats(), "cache": self.runner.stats() if self.runner else {}}
