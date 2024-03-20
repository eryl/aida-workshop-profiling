from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    batch_size: int = 32
    max_epochs: int = 10
    learning_rate: float = 1e-3
    update_iterations: int = 10