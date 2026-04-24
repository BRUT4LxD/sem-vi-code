"""
Single entry point for the ImageNette full-research pipeline.

Run from repository root::

    python -m experiments.imagenette_full_research.runner --config experiments/imagenette_full_research/config.yaml

Or::

    python experiments/imagenette_full_research/runner.py
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from attacks.attack_names import AttackNames
from config.imagenet_models import ImageNetModels
from data_eng.dataset_loader import load_imagenette
from domain.model.model_names import ModelNames
from experiments.imagenette_full_research.paths import FullResearchPaths, load_yaml_dict, paths_from_mapping
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
    arch = stem.split("_")[0]
    for name in ModelNames().all_model_names:
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


def run(config: Dict[str, Any], paths: FullResearchPaths, resume: bool, phases: Optional[Sequence[str]]) -> None:
    paths.ensure_dirs()
    train_cfg = config.get("training", {})
    prog_cfg = config.get("progressive", {})
    pas_cfg = config.get("passive_adversarial", {})
    noise_cfg = config.get("noise_detection", {})
    trans_cfg = config.get("transferability", {})

    config_name = train_cfg.get("config_name", ImageNetteTrainingConfigs.ADVANCED)
    full_finetune = bool(train_cfg.get("full_finetune", True))

    archs = _default_architectures()
    attack_names = AttackNames().all_attack_names

    def want(phase: str) -> bool:
        return phases is None or phase in phases

    # --- Phase 1: baseline training ---
    # if want("train_baseline"):
    #     trainer = ImageNetteStandardTrainer(
    #         device="auto",
    #         models_dir=paths.models_normal,
    #         tensorboard_runs_root=paths.runs,
    #     )
    #     for arch in archs:
    #         save_path = os.path.join(paths.models_normal, f"{arch}_{config_name}.pt")
    #         if resume and os.path.isfile(save_path):
    #             log.info("Skipping baseline train (exists): %s", save_path)
    #             continue
    #         log.info("Baseline training: %s", arch)
    #         trainer.train_model(
    #             arch,
    #             config_name=config_name,
    #             full_finetune=full_finetune,
    #         )

    # # --- Clean validation (normal) ---
    # if want("validate_normal"):
    #     val = ImageNetteValidator(
    #         models_dir=paths.models_normal,
    #         results_dir=paths.results_normal,
    #         device="auto",
    #     )
    #     tuples = []
    #     for ck in glob.glob(os.path.join(paths.models_normal, "*.pt")):
    #         tuples.append(_checkpoint_tuple(ck))
    #     if tuples:
    #         results = val.validate_models_from_tuples(tuples)
    #         val.save_models_from_tuples_summary(results, output_filename="clean_validation_summary.csv")

    # --- Generation phase removed by request ---
    if want("attacks_normal"):
        log.info("Skipping attacks_normal: generation step removed (direct attacks run in-memory).")

    # --- Direct attacks in-memory (normal) ---
    if want("direct_normal"):
        _, test_loader = load_imagenette(batch_size=8, test_subset_size=-1)
        da = ImageNetteDirectAttacks(device="auto")

        def ck_normal(name: str) -> str:
            return os.path.join(paths.models_normal, f"{name}_{config_name}.pt")

        new_attack_names = [
            AttackNames().SPSA,
            AttackNames().TIFGSM,
            AttackNames().TPGD,
            AttackNames().UPGD,
            AttackNames().VMIFGSM,
            AttackNames().VNIFGSM,
            AttackNames().OnePixel,
            AttackNames().Pixle,
            AttackNames().Square,
        ]

        extra_attack_models = [ModelNames().resnet18]

        da.run_attacks_on_models(
            attack_names=new_attack_names,
            data_loader=test_loader,
            model_names=extra_attack_models,
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
            m = ImageNetModels.get_model(arch)
            m.__class__.__name__ = arch
            m = SetupPretraining.setup_imagenette(m, full_finetune=True)
            models.append(m)
            stem = f"{arch}_progressive_adv"
            save_paths.append(os.path.join(paths.models_progressive_active, f"{stem}.pt"))
            folder_names.append(stem)

        _vip = prog_cfg.get("validation_images_per_attack_per_iteration", None)
        validation_images_per_attack = (
            int(_vip) if _vip is not None else None
        )
        prog_trainer.train_multiple_progressive_adversarial_models(
            models=models,
            attack_names=attack_names,
            learning_rate=float(prog_cfg.get("learning_rate", 0.001)),
            iterations=int(prog_cfg.get("iterations", 5)),
            epochs_per_iteration=int(prog_cfg.get("epochs_per_iteration", 2)),
            batch_size=int(prog_cfg.get("batch_size", 32)),
            images_per_attack_per_iteration=int(
                prog_cfg.get("images_per_attack_per_iteration", 10)
            ),
            validation_images_per_attack_per_iteration=validation_images_per_attack,
            early_stopping_patience=int(prog_cfg.get("early_stopping_patience", 7)),
            scheduler_type=str(prog_cfg.get("scheduler_type", "step")),
            weight_decay=float(prog_cfg.get("weight_decay", 0.0001)),
            gradient_clip_norm=float(prog_cfg.get("gradient_clip_norm", 1.0)),
            save_generated_images=bool(prog_cfg.get("save_generated_images", True)),
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

    # --- Generation phase removed by request ---
    if want("attacks_progressive_active"):
        log.info("Skipping attacks_progressive_active: generation step removed (direct attacks run in-memory).")

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
                use_preattacked_images=True,
                batch_size=int(pas_cfg.get("batch_size", 32)),
                num_epochs=int(pas_cfg.get("num_epochs", 15)),
                learning_rate=float(pas_cfg.get("learning_rate", 0.001)),
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

    # --- Generation phase removed by request ---
    if want("attacks_passive"):
        log.info("Skipping attacks_passive: generation step removed (direct attacks run in-memory).")

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
                batch_size=int(noise_cfg.get("batch_size", 32)),
                learning_rate=float(noise_cfg.get("learning_rate", 0.001)),
                num_epochs=int(noise_cfg.get("num_epochs", 20)),
                attacked_subset_size=int(noise_cfg.get("attacked_subset_size", 10000)),
                clean_to_attacked_ratio=float(noise_cfg.get("clean_to_attacked_ratio", 1.0)),
                augment_clean_to_match_attacked=bool(
                    noise_cfg.get("augment_clean_to_match_attacked", True)
                ),
                early_stopping_patience=int(noise_cfg.get("early_stopping_patience", 10)),
                scheduler_type=str(noise_cfg.get("scheduler_type", "plateau")),
                weight_decay=float(noise_cfg.get("weight_decay", 0.0001)),
                gradient_clip_norm=float(noise_cfg.get("gradient_clip_norm", 1.0)),
                save_model_path=save_p,
            )

    # --- Transferability (from files) ---
    if want("transferability") and bool(trans_cfg.get("enabled", True)):
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ImageNette full research runner")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML/JSON file with paths + phase settings",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip training steps when output checkpoints already exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated phases to run (default: all). See runner.py for names.",
    )
    args = parser.parse_args(argv)

    raw = load_yaml_dict(args.config) if args.config else {}
    if isinstance(raw, dict) and "paths" in raw:
        path_dict = raw.get("paths") or {}
    else:
        path_dict = {}
    paths = paths_from_mapping(path_dict if isinstance(path_dict, dict) else {})
    cfg: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

    phases = [p.strip() for p in args.only.split(",") if p.strip()] if args.only else None

    run(cfg, paths, resume=args.resume, phases=phases)


if __name__ == "__main__":
    main()
