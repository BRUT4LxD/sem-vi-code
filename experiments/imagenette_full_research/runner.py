"""
Single entry point for the ImageNette full-research pipeline.

Run from repository root::

    python -m experiments.imagenette_full_research.runner

Or::

    python experiments/imagenette_full_research/runner.py
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from typing import List, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

from attacks.attack_names import AttackNames
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import load_imagenette
from data_eng.io import load_model_imagenette
from domain.model.model_names import ModelNames
from experiments.imagenette_full_research.pipeline_config import (
    FullResearchConfig,
    load_full_research_config,
)
from imagenette_lab.imagenette_direct_attacks import ImageNetteDirectAttacks
from imagenette_lab.imagenette_transferability_attacks import imagenette_transferability_model2model_from_files
from imagenette_lab.imagenette_validator import ImageNetteValidator
from imagenette_lab.training.imagenette_adversarial_progressive_trainer import (
    ImageNetteAdversarialProgressiveTrainer,
)
from imagenette_lab.training.imagenette_adversarial_trainer import ImageNetteAdversarialTrainer
from imagenette_lab.training.imagenette_noise_detection_trainer import ImageNetteNoiseDetectionTrainer
from imagenette_lab.training.imagenette_standard_trainer import ImageNetteStandardTrainer
from imagenette_lab.training.imagenette_training_configs import ImageNetteTrainingConfigs
from training.transfer.setup_pretraining import SetupPretraining

log = logging.getLogger("imagenette_full_research")


def _default_architectures() -> List[str]:
    return list(ImageNetteTrainingConfigs.AVAILABLE_MODELS)


def _stem(path: str) -> str:
    base = os.path.basename(path)
    if base.lower().endswith(".pt"):
        return os.path.splitext(base)[0]
    return base


def _checkpoint_tuple(ckpt_path: str) -> Tuple[str, str]:
    """``(architecture_hint, path)`` for validation / loading."""
    stem = _stem(ckpt_path)
    arch = stem
    for name in sorted(ModelNames().all_model_names, key=len, reverse=True):
        if stem.startswith(name):
            arch = name
            break
    return arch, os.path.abspath(ckpt_path)


def _transfer_tuples_from_dir(models_dir: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for ckpt_path in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
        stem = _stem(ckpt_path)
        out.append((stem, os.path.abspath(ckpt_path)))
    return out


def _validated_attack_names(raw_attack_names: List[str]) -> List[str]:
    attacks = AttackNames()
    allowed = set(attacks.all_attack_names)

    if not raw_attack_names:
        raise ValueError("attacks.names must contain at least one attack")

    for attack_name in raw_attack_names:
        if attack_name not in allowed:
            raise ValueError(
                f"attacks.names: unknown attack {attack_name!r}. "
                f"Allowed: {sorted(allowed)}"
            )

    return raw_attack_names


def run(config: FullResearchConfig) -> None:
    paths = config.paths
    paths.ensure_dirs()
    phases = config.run.phases

    config_name = config.training.config_name
    full_finetune = config.training.full_finetune
    resume = config.run.resume

    _allowed = set(ImageNetteTrainingConfigs.AVAILABLE_MODELS)
    if config.training.architectures:
        archs = config.training.architectures
        for a in archs:
            if a not in _allowed:
                raise ValueError(
                    f"training.architectures: unknown model {a!r}. "
                    f"Allowed: {sorted(_allowed)}"
                )
    else:
        archs = _default_architectures()
    attack_names = _validated_attack_names(config.attacks.names)

    def want(phase: str) -> bool:
        return not phases or phase in phases

    # --- Phase 1: baseline training ---
    if want("train_baseline") :
        trainer = ImageNetteStandardTrainer(
            device="auto",
            models_dir=paths.models_normal,
            tensorboard_runs_root=paths.runs,
        )
        for arch in archs:
            save_path = os.path.join(paths.models_normal, f"{arch}_{config_name}.pt")
            if resume and os.path.isfile(save_path):
                log.info("Skipping baseline train (exists): %s", save_path)
                continue
            log.info("Baseline training: %s", arch)
            trainer.train_model(
                arch,
                config_name=config_name,
                full_finetune=full_finetune,
            )

    # --- Clean validation (normal) ---
    if want("validate_normal"):
        val = ImageNetteValidator(
            models_dir=paths.models_normal,
            results_dir=paths.results_normal,
            device="auto",
        )
        tuples = []
        for ck in glob.glob(os.path.join(paths.models_normal, "*.pt")):
            tuples.append(_checkpoint_tuple(ck))
        if tuples:
            results = val.validate_models_from_tuples(tuples)
            val.save_models_from_tuples_summary(results, output_filename="clean_validation_summary.csv")
            for result in results:
                val.save_model_per_class_results(result)

    # --- Direct attacks in-memory (normal) ---
    if want("direct_normal"):
        _, test_loader = load_imagenette(batch_size=1, test_subset_size=500)
        da = ImageNetteDirectAttacks(device="auto")

        def ck_normal(name: str) -> str:
            return os.path.join(paths.models_normal, f"{name}_{config_name}.pt")

        da.run_attacks_on_models(
            attack_names=attack_names,
            data_loader=test_loader,
            model_names=archs,
            results_folder=paths.results_attacks_normal,
            checkpoint_path_for_model=ck_normal,
        )

    # --- Progressive active ---
    if want("progressive_active"):
        prog_trainer = ImageNetteAdversarialProgressiveTrainer(
            device="auto",
            models_dir=paths.models_normal,
            progressive_adversarial_models_dir=paths.models_progressive_active,
            tensorboard_runs_root=paths.runs,
        )
        models: List = []
        save_paths: List[str] = []
        folder_names: List[str] = []
        for arch in archs:
            normal_checkpoint = os.path.join(paths.models_normal, f"{arch}_{config_name}.pt")
            loaded = load_model_imagenette(
                normal_checkpoint,
                model_name=arch,
                device=str(prog_trainer.device),
                verbose=False,
            )
            if not loaded.success:
                raise RuntimeError(
                    f"Failed to load normal model for progressive_active: "
                    f"{normal_checkpoint}. {loaded.error}"
                )
            models.append(loaded.model)
            stem = f"{arch}_progressive_adv"
            save_paths.append(os.path.join(paths.models_progressive_active, f"{stem}.pt"))
            folder_names.append(stem)

        prog_trainer.train_multiple_progressive_adversarial_models(
            models=models,
            attack_names=attack_names,
            learning_rate=config.progressive.learning_rate,
            iterations=config.progressive.iterations,
            epochs_per_iteration=config.progressive.epochs_per_iteration,
            batch_size=config.progressive.batch_size,
            images_per_attack_per_iteration=(
                config.progressive.images_per_attack_per_iteration
            ),
            validation_images_per_attack_per_iteration=(
                config.progressive.validation_images_per_attack_per_iteration
            ),
            early_stopping_patience=config.progressive.early_stopping_patience,
            scheduler_type=config.progressive.scheduler_type,
            weight_decay=config.progressive.weight_decay,
            gradient_clip_norm=config.progressive.gradient_clip_norm,
            save_generated_images=config.progressive.save_generated_images,
            attacked_images_folder=paths.data_attacks_progressive_active,
            saved_attack_folder_names=folder_names,
            save_model_paths=save_paths,
        )

    # --- Validate progressive active ---
    if want("validate_progressive_active"):
        val = ImageNetteValidator(
            models_dir=paths.models_progressive_active,
            results_dir=paths.results_progressive_active,
            device="auto",
        )
        tuples = []
        for ck in glob.glob(os.path.join(paths.models_progressive_active, "*.pt")):
            tuples.append(_checkpoint_tuple(ck))
        if tuples:
            results = val.validate_models_from_tuples(tuples)
            val.save_models_from_tuples_summary(results, output_filename="clean_validation_summary.csv")

    if want("direct_progressive_active"):
        _, test_loader = load_imagenette(batch_size=8, test_subset_size=-1)
        da = ImageNetteDirectAttacks(device="auto")

        def ck_prog(name: str) -> str:
            return os.path.join(paths.models_progressive_active, f"{name}_progressive_adv.pt")

        da.run_attacks_on_models(
            attack_names=attack_names,
            data_loader=test_loader,
            model_names=archs,
            results_folder=paths.results_attacks_progressive_active,
            checkpoint_path_for_model=ck_prog,
        )

    # --- Passive adversarial (train on progressive-active disk attacks) ---
    if want("passive"):
        adv_trainer = ImageNetteAdversarialTrainer(
            device="auto",
            models_dir=paths.models_progressive_passive,
            adversarial_models_dir=paths.models_progressive_passive,
            tensorboard_runs_root=paths.runs,
        )
        for arch in archs:
            save_p = os.path.join(paths.models_progressive_passive, f"{arch}_adv_passive.pt")
            if resume and os.path.isfile(save_p):
                log.info("Skipping passive train (exists): %s", save_p)
                continue
            model = ImageNetModels.get_model(arch)
            model.__class__.__name__ = arch
            model = SetupPretraining.setup_imagenette(model, full_finetune=True)
            log.info("Passive adversarial training: %s", arch)
            adv_trainer.train_adversarial_model(
                model=model,
                attack_names=None,
                use_preattacked_images=(
                    config.passive_adversarial.use_preattacked_images
                ),
                batch_size=config.passive_adversarial.batch_size,
                num_epochs=config.passive_adversarial.num_epochs,
                learning_rate=config.passive_adversarial.learning_rate,
                adversarial_ratio=config.passive_adversarial.adversarial_ratio,
                early_stopping_patience=(
                    config.passive_adversarial.early_stopping_patience
                ),
                scheduler_type=config.passive_adversarial.scheduler_type,
                weight_decay=config.passive_adversarial.weight_decay,
                gradient_clip_norm=config.passive_adversarial.gradient_clip_norm,
                attacked_subset_size=config.passive_adversarial.attacked_subset_size,
                augment_clean_to_match_attacked=(
                    config.passive_adversarial.augment_clean_to_match_attacked
                ),
                train_test_split=config.passive_adversarial.train_test_split,
                attacked_images_folder=paths.data_attacks_progressive_active,
                clean_train_root=paths.imagenette_train,
                clean_val_root=paths.imagenette_val,
                save_model_path=save_p,
            )

    # --- Validate passive ---
    if want("validate_passive"):
        val = ImageNetteValidator(
            models_dir=paths.models_progressive_passive,
            results_dir=paths.results_progressive_passive,
            device="auto",
        )
        tuples = []
        for ck in glob.glob(os.path.join(paths.models_progressive_passive, "*.pt")):
            tuples.append(_checkpoint_tuple(ck))
        if tuples:
            results = val.validate_models_from_tuples(tuples)
            val.save_models_from_tuples_summary(results, output_filename="clean_validation_passive_summary.csv")

    if want("direct_passive"):
        _, test_loader = load_imagenette(batch_size=8, test_subset_size=-1)
        da = ImageNetteDirectAttacks(device="auto")

        def ck_pass(name: str) -> str:
            return os.path.join(paths.models_progressive_passive, f"{name}_adv_passive.pt")

        da.run_attacks_on_models(
            attack_names=attack_names,
            data_loader=test_loader,
            model_names=archs,
            results_folder=paths.results_attacks_progressive_passive,
            checkpoint_path_for_model=ck_pass,
        )

    # --- Noise detection (merged attacked roots) ---
    if want("noise"):
        nd = ImageNetteNoiseDetectionTrainer(
            device="auto",
            models_dir=paths.models_noise_detection,
            noise_detection_dir=paths.models_noise_detection,
            tensorboard_runs_root=paths.runs,
        )
        attacked_roots = [paths.data_attacks_normal, paths.data_attacks_progressive_active]
        for arch in archs:
            save_p = os.path.join(paths.models_noise_detection, f"{arch}_noise_detector.pt")
            if resume and os.path.isfile(save_p):
                log.info("Skipping noise detector (exists): %s", save_p)
                continue
            nd.train_noise_detection_model(
                model_name=arch,
                attacked_images_folder=attacked_roots,
                clean_train_folder=paths.imagenette_train,
                clean_test_folder=paths.imagenette_val,
                batch_size=config.noise_detection.batch_size,
                learning_rate=config.noise_detection.learning_rate,
                num_epochs=config.noise_detection.num_epochs,
                attacked_subset_size=config.noise_detection.attacked_subset_size,
                clean_to_attacked_ratio=(
                    config.noise_detection.clean_to_attacked_ratio
                ),
                augment_clean_to_match_attacked=(
                    config.noise_detection.augment_clean_to_match_attacked
                ),
                early_stopping_patience=(
                    config.noise_detection.early_stopping_patience
                ),
                scheduler_type=config.noise_detection.scheduler_type,
                weight_decay=config.noise_detection.weight_decay,
                gradient_clip_norm=config.noise_detection.gradient_clip_norm,
                save_model_path=save_p,
            )

    # --- Transferability (from files) ---
    if want("transferability") and config.transferability.enabled:
        models_n = _transfer_tuples_from_dir(paths.models_normal)
        if models_n:
            imagenette_transferability_model2model_from_files(
                models=models_n,
                attack_names=attack_names,
                attacked_images_folder=paths.data_attacks_normal,
                results_folder=paths.results_transferability_normal,
            )
        models_p = _transfer_tuples_from_dir(paths.models_progressive_passive)
        if models_p:
            imagenette_transferability_model2model_from_files(
                models=models_p,
                attack_names=attack_names,
                attacked_images_folder=paths.data_attacks_progressive_passive,
                results_folder=paths.results_transferability_passive,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) > 1:
        raise SystemExit("This runner does not accept CLI arguments. Edit config.yaml instead.")

    config = load_full_research_config(_CONFIG_PATH)
    run(config)


if __name__ == "__main__":
    main()
