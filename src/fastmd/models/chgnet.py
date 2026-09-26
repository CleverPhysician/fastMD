"""CHGNet adapter: captured energy/forces, eager stress and magnetic moments."""
from ase import units

from .base import ModelBackend, ModelCapabilities


class CHGNetModel(ModelBackend):
    capabilities = ModelCapabilities(
        frozenset({"energy", "forces", "stress", "magmoms"}),
        frozenset({"energy", "forces"}),
    )

    def __init__(self, *, checkpoint=None, model_name="0.3.0", **kwargs):
        super().__init__(**kwargs)
        from fastmd._vendor.chgnet.model.model import CHGNet
        self.model = (CHGNet.from_file(str(checkpoint)) if checkpoint is not None
                      else CHGNet.load(model_name=model_name, use_device=str(self.device), verbose=False))
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.runner = None

    def graph_unavailable_reason(self, properties):
        reason = super().graph_unavailable_reason(properties)
        if reason:
            return reason
        if not self.model.mlp_first:
            return "CHGNet CUDA Graph requires mlp_first=True"
        from fastmd._vendor.chgnet.graph.gpu_graph_builder import op_available
        return None if op_available else "GPU neighbor operators missing; install fastMD[cuda]"

    def _predict_eager(self, atoms, properties):
        from pymatgen.io.ase import AseAtomsAdaptor
        task = "efsm" if "magmoms" in properties else ("efs" if "stress" in properties else "ef")
        graph = self.model.graph_converter(AseAtomsAdaptor.get_structure(atoms))
        output = self.model.predict_graph(graph.to(self.device), task=task)
        scale = len(atoms) if self.model.is_intensive else 1
        results = {"energy": output["e"] * scale, "forces": output["f"]}
        if "s" in output:
            results["stress"] = output["s"] * units.GPa
        if "m" in output:
            results["magmoms"] = output["m"]
        return results

    def _predict_graph(self, atoms, properties):
        from fastmd._vendor.chgnet.graph.gpu_graph_builder import atoms_to_graph_gpu
        from fastmd._vendor.chgnet.model.cuda_graph import BucketedGraphRunner
        from fastmd._vendor.chgnet.model.triton_fusions import model_fusion_mode, prepare_model_fusions
        if self.runner is None:
            if self.config.enable_fusions:
                prepare_model_fusions(self.model)
            self.runner = BucketedGraphRunner(self.model, u_step=self.config.edge_capacity_step or 128,
                                             t_step=self.config.triplet_capacity_step or 1024,
                                             warmup=self.config.warmup_steps)
        converter = self.model.graph_converter
        graph = atoms_to_graph_gpu(atoms, atom_graph_cutoff=converter.atom_graph_cutoff,
                                   bond_graph_cutoff=converter.bond_graph_cutoff, device=self.device)
        with model_fusion_mode(self.config.enable_fusions):
            output, n = self.runner.run(graph)
        scale = n if self.model.is_intensive else 1
        results = {"energy": float(output["e"][0].detach()) * scale,
                   "forces": output["f"][0][:n].detach().cpu().numpy().copy()}
        if len(self.runner.cache) > self.config.max_cached_graphs:
            self.clear_graphs()
        return results

    def clear_graphs(self):
        self.runner = None

    def stats(self):
        return {**super().stats(), "cache": self.runner.stats() if self.runner else {}}
