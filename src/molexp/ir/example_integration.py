from pydantic import BaseModel, Field
from molexp.task_base import Task
from molexp.ir.registry import registry

# 1. Define Config
class GeometryOptimizationConfig(BaseModel):
    method: str = "B3LYP"
    basis_set: str = "6-31G*"
    max_cycles: int = Field(100, ge=1)

# 2. Define Task & Register
@registry.register("chemistry.optimize_geometry", GeometryOptimizationConfig)
class OptimizeGeometryTask(Task):
    cfg_model = GeometryOptimizationConfig
    
    def forward(self, *args, cfg: GeometryOptimizationConfig):
        # In a real task, args would be upstream data (e.g. molecule structure)
        input_data = args[0] if args else "no_input"
        print(f"Optimizing {input_data} with {cfg.method}/{cfg.basis_set}")
        return f"optimized_{input_data}"

# Example of a data producer
class LoadMoleculeConfig(BaseModel):
    path: str

@registry.register("io.load_molecule", LoadMoleculeConfig)
class LoadMoleculeTask(Task):
    cfg_model = LoadMoleculeConfig
    
    def forward(self, cfg: LoadMoleculeConfig):
        print(f"Loading molecule from {cfg.path}")
        return f"molecule_from_{cfg.path}"
