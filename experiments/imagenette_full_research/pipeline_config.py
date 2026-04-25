from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from experiments.imagenette_full_research.paths import (
    FullResearchPaths,
    load_yaml_dict,
    paths_from_mapping,
)
from imagenette_lab.training.imagenette_training_configs import ImageNetteTrainingConfigs


@dataclass
class RunConfig:
    resume: bool = False
    phases: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    config_name: str = ImageNetteTrainingConfigs.ADVANCED
    full_finetune: bool = True
    architectures: List[str] = field(default_factory=list)


@dataclass
class AttacksConfig:
    names: List[str] = field(default_factory=list)
    images_per_class: int = 50
    run_train_split: bool = True
    run_test_split: bool = True


@dataclass
class ProgressiveConfig:
    learning_rate: float = 0.001
    iterations: int = 5
    epochs_per_iteration: int = 2
    batch_size: int = 32
    images_per_attack_per_iteration: int = 10
    validation_images_per_attack_per_iteration: Optional[int] = None
    early_stopping_patience: int = 7
    scheduler_type: str = "step"
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0
    save_generated_images: bool = True


@dataclass
class PassiveAdversarialConfig:
    use_preattacked_images: bool = True
    attacked_subset_size: int = 20000
    augment_clean_to_match_attacked: bool = True
    train_test_split: Optional[float] = 0.2
    learning_rate: float = 0.001
    num_epochs: int = 1000
    batch_size: int = 128
    adversarial_ratio: float = 0.5
    early_stopping_patience: int = 7
    scheduler_type: str = "step"
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


@dataclass
class NoiseDetectionConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 20
    attacked_subset_size: int = 10000
    clean_to_attacked_ratio: float = 1.0
    augment_clean_to_match_attacked: bool = True
    early_stopping_patience: int = 10
    scheduler_type: str = "plateau"
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 1.0


@dataclass
class TransferabilityConfig:
    enabled: bool = True


@dataclass
class FullResearchConfig:
    paths: FullResearchPaths = field(default_factory=FullResearchPaths)
    run: RunConfig = field(default_factory=RunConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attacks: AttacksConfig = field(default_factory=AttacksConfig)
    progressive: ProgressiveConfig = field(default_factory=ProgressiveConfig)
    passive_adversarial: PassiveAdversarialConfig = field(
        default_factory=PassiveAdversarialConfig
    )
    noise_detection: NoiseDetectionConfig = field(default_factory=NoiseDetectionConfig)
    transferability: TransferabilityConfig = field(default_factory=TransferabilityConfig)


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _string_list(section: Dict[str, Any], key: str, section_name: str) -> List[str]:
    value = section.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{section_name}.{key} must be a YAML list")
    return [str(item).strip() for item in value if str(item).strip()]


def load_full_research_config(path: str) -> FullResearchConfig:
    raw = load_yaml_dict(path)
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a top-level YAML mapping")

    run = _section(raw, "run")
    training = _section(raw, "training")
    attacks = _section(raw, "attacks")
    progressive = _section(raw, "progressive")
    passive = _section(raw, "passive_adversarial")
    noise = _section(raw, "noise_detection")
    transferability = _section(raw, "transferability")
    paths = _section(raw, "paths")

    return FullResearchConfig(
        paths=paths_from_mapping(paths),
        run=RunConfig(
            resume=bool(run.get("resume", False)),
            phases=_string_list(run, "phases", "run"),
        ),
        training=TrainingConfig(
            config_name=str(
                training.get("config_name", ImageNetteTrainingConfigs.ADVANCED)
            ),
            full_finetune=bool(training.get("full_finetune", True)),
            architectures=_string_list(training, "architectures", "training"),
        ),
        attacks=AttacksConfig(
            names=_string_list(attacks, "names", "attacks"),
            images_per_class=int(attacks.get("images_per_class", 50)),
            run_train_split=bool(attacks.get("run_train_split", True)),
            run_test_split=bool(attacks.get("run_test_split", True)),
        ),
        progressive=ProgressiveConfig(
            learning_rate=float(progressive.get("learning_rate", 0.001)),
            iterations=int(progressive.get("iterations", 5)),
            epochs_per_iteration=int(progressive.get("epochs_per_iteration", 2)),
            batch_size=int(progressive.get("batch_size", 32)),
            images_per_attack_per_iteration=int(
                progressive.get("images_per_attack_per_iteration", 10)
            ),
            validation_images_per_attack_per_iteration=(
                int(progressive["validation_images_per_attack_per_iteration"])
                if progressive.get("validation_images_per_attack_per_iteration")
                is not None
                else None
            ),
            early_stopping_patience=int(
                progressive.get("early_stopping_patience", 7)
            ),
            scheduler_type=str(progressive.get("scheduler_type", "step")),
            weight_decay=float(progressive.get("weight_decay", 0.0001)),
            gradient_clip_norm=float(progressive.get("gradient_clip_norm", 1.0)),
            save_generated_images=bool(
                progressive.get("save_generated_images", True)
            ),
        ),
        passive_adversarial=PassiveAdversarialConfig(
            use_preattacked_images=bool(passive.get("use_preattacked_images", True)),
            attacked_subset_size=int(passive.get("attacked_subset_size", 20000)),
            augment_clean_to_match_attacked=bool(
                passive.get("augment_clean_to_match_attacked", True)
            ),
            train_test_split=(
                float(passive["train_test_split"])
                if passive.get("train_test_split") is not None
                else None
            ),
            learning_rate=float(passive.get("learning_rate", 0.001)),
            num_epochs=int(passive.get("num_epochs", 1000)),
            batch_size=int(passive.get("batch_size", 128)),
            adversarial_ratio=float(passive.get("adversarial_ratio", 0.5)),
            early_stopping_patience=int(passive.get("early_stopping_patience", 7)),
            scheduler_type=str(passive.get("scheduler_type", "step")),
            weight_decay=float(passive.get("weight_decay", 0.0001)),
            gradient_clip_norm=float(passive.get("gradient_clip_norm", 1.0)),
        ),
        noise_detection=NoiseDetectionConfig(
            batch_size=int(noise.get("batch_size", 32)),
            learning_rate=float(noise.get("learning_rate", 0.001)),
            num_epochs=int(noise.get("num_epochs", 20)),
            attacked_subset_size=int(noise.get("attacked_subset_size", 10000)),
            clean_to_attacked_ratio=float(noise.get("clean_to_attacked_ratio", 1.0)),
            augment_clean_to_match_attacked=bool(
                noise.get("augment_clean_to_match_attacked", True)
            ),
            early_stopping_patience=int(noise.get("early_stopping_patience", 10)),
            scheduler_type=str(noise.get("scheduler_type", "plateau")),
            weight_decay=float(noise.get("weight_decay", 0.0001)),
            gradient_clip_norm=float(noise.get("gradient_clip_norm", 1.0)),
        ),
        transferability=TransferabilityConfig(
            enabled=bool(transferability.get("enabled", True))
        ),
    )
