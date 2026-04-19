import os
import csv
import gc
import time
import datetime
import traceback
import torch
from typing import Callable, List, Dict, Tuple, Optional, Set, NamedTuple, TypeVar
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

from attacks.attack_factory import AttackFactory
from attacks.attack_names import AttackNames
from data_eng.io import load_model_imagenette
from data_eng.dataset_loader import load_imagenette
from domain.model.model_names import ModelNames
from shared.model_utils import ModelUtils


T = TypeVar("T")


def _is_allocation_error(exc: BaseException) -> bool:
    """True for CUDA OOM / memory allocation failures that may clear on retry."""
    if isinstance(exc, MemoryError):
        return True
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    accel_cls = getattr(torch, "AcceleratorError", None)
    if accel_cls is not None and isinstance(exc, accel_cls):
        msg = str(exc).lower()
        if "memory" in msg or "out of memory" in msg or "cudaerror" in msg:
            return True
    msg = str(exc).lower()
    if "out of memory" in msg or "cudaerrormemoryallocation" in msg:
        return True
    if "cuda" in msg and "alloc" in msg:
        return True
    return False


def _is_clear_cuda_oom_message(exc: BaseException) -> bool:
    """Explicit OOM text — always eligible for allocation retry (ignores short-run threshold)."""
    t = str(exc).lower()
    return "out of memory" in t or "cudaerrormemoryallocation" in t


def _defragment_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# Small batches on target forward reduce activation VRAM (e.g. VGG with many adv images).
_TARGET_TRANSFER_MICRO_BATCH: int = 16


def run_transferability_with_allocation_retry(
    run_fn: Callable[[], T],
    *,
    min_elapsed_seconds_to_retry: float = 60.0,
    max_allocation_retries: int = 50,
) -> T:
    """
    Run ``run_fn()`` in a loop. On allocation / CUDA OOM errors:

    * If the attempt lasted **at least** ``min_elapsed_seconds_to_retry`` seconds, clear GPU
      cache (if available) and **retry** (up to ``max_allocation_retries`` times).
    * If the attempt lasted **less** than that threshold, **do not retry** — re-raise
      (avoids spinning on failures that are unlikely to be transient fragmentation).

    Other exceptions are always re-raised immediately.
    """
    allocation_failures = 0
    while True:
        t0 = time.perf_counter()
        try:
            return run_fn()
        except Exception as e:
            elapsed = time.perf_counter() - t0
            if not _is_allocation_error(e):
                raise
            # Fast OOM (e.g. one huge forward on VGG) should still retry; threshold keeps other
            # allocation errors from spinning when the run barely started.
            if (
                elapsed < min_elapsed_seconds_to_retry
                and not _is_clear_cuda_oom_message(e)
            ):
                print(
                    f"⚠️ Allocation error after {elapsed:.1f}s "
                    f"(<{min_elapsed_seconds_to_retry:.0f}s threshold); not retrying."
                )
                raise
            allocation_failures += 1
            if allocation_failures > max_allocation_retries:
                print(
                    f"⚠️ Allocation errors exceeded max_allocation_retries="
                    f"{max_allocation_retries}; giving up."
                )
                raise
            print(
                f"⚠️ Allocation error after {elapsed:.1f}s "
                f"(attempt recovery {allocation_failures}/{max_allocation_retries}); "
                f"clearing cache and retrying…"
            )
            _defragment_cuda_memory()


class ModelCheckpoint(NamedTuple):
    """Resolved checkpoint: folder label (tuple first element), absolute path, torchvision architecture key."""

    label: str
    path: str
    architecture: str


def _checkpoint_csv_name(model_path: str) -> str:
    """Checkpoint file stem (basename without ``.pt``); written to CSV ``source_model`` / ``target_model``."""
    base = os.path.basename(model_path)
    root, ext = os.path.splitext(base)
    return root if ext.lower() == ".pt" else base


def _infer_architecture_from_path(model_path: str) -> str:
    """
    Infer ``ModelNames`` architecture string from the basename (substring match, longest wins).
    Filenames should contain e.g. ``resnet18``, ``densenet121``, ``efficientnet_b0``.
    """
    base = os.path.basename(model_path).lower()
    candidates = sorted(ModelNames().all_model_names, key=len, reverse=True)
    for name in candidates:
        if name.lower() in base:
            return name
    raise ValueError(
        f"Could not infer model architecture from filename {model_path!r}. "
        "Include a known architecture substring (e.g. resnet18, densenet121) in the .pt name, "
        "or use a known architecture as the first tuple element (model_name)."
    )


def _architecture_for_checkpoint(model_name: str, model_path: str) -> str:
    """If ``model_name`` is a known torchvision key, use it; else infer from ``model_path``."""
    mn = model_name.strip()
    if mn in ModelNames().all_model_names:
        return mn
    return _infer_architecture_from_path(model_path)


def normalize_model_tuples(models: List[Tuple[str, str]]) -> List[ModelCheckpoint]:
    """
    Validate ``models`` as ``(model_name, model_path)`` pairs.

    ``model_name`` is the label used for adversarial-image folder names (from-files mode);
    CSV columns ``source_model`` / ``target_model`` use the checkpoint stem (basename without ``.pt``; see :func:`_checkpoint_csv_name`).
    ``model_path`` is the ``.pt`` file to load. Architecture for ``load_model_imagenette`` is
    ``model_name`` when it is a known :class:`ModelNames` value; otherwise it is inferred from
    the file path.
    """
    if not models:
        raise ValueError("models must be a non-empty list of (model_name, model_path) tuples")
    out: List[ModelCheckpoint] = []
    for model_name, model_path in models:
        ap = os.path.abspath(os.path.normpath(model_path))
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"Model file not found: {ap}")
        label = model_name.strip()
        arch = _architecture_for_checkpoint(label, ap)
        out.append(ModelCheckpoint(label=label, path=ap, architecture=arch))
    return out


def _resolve_saved_adv_images_dir(
    attacked_images_folder: str, source_model_name: str, attack_name: str
) -> Optional[str]:
    """
    Resolve directory with saved adversarial PNGs.

    Supports layout from imagenette_adv_imgs_generator:
        {root}/{train|test}/{model}/{attack}/{class}/*.png
    and legacy flat layout:
        {root}/{model}/{attack}/{class}/*.png
    """
    candidates = [
        os.path.join(attacked_images_folder, "train", source_model_name, attack_name),
        os.path.join(attacked_images_folder, "test", source_model_name, attack_name),
        os.path.join(attacked_images_folder, source_model_name, attack_name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _resolve_model_adv_images_dir(
    attacked_images_folder: str, source_label: str, attack_name: str
) -> Optional[str]:
    """Prefer ``{root}/{source_label}/{attack}``; else same layouts as :func:`_resolve_saved_adv_images_dir`."""
    flat = os.path.join(attacked_images_folder, source_label, attack_name)
    if os.path.isdir(flat):
        return flat
    return _resolve_saved_adv_images_dir(attacked_images_folder, source_label, attack_name)


# Short CSV basename tags for incremental / bulk CSV (filename_key, method) -> prefix
_TRANSFERABILITY_CSV_PREFIX: Dict[Tuple[str, str], str] = {
    ("model2model_transferability", "in_memory"): "m2m_trans_mem",
    ("attack2model_transferability", "in_memory"): "a2m_trans_mem",
    ("model2model_transferability", "from_files"): "m2m_trans_files",
    ("attack2model_transferability", "from_files"): "a2m_trans_files",
}

TRANSFERABILITY_CSV_FIELDNAMES: List[str] = [
    "source_model",
    "target_model",
    "attack_name",
    "total_images",
    "total_successful_attacks",
    "transfer_success",
    "transfer_rate",
    "attack_success_rate",
    "timestamp",
]


def _transferability_csv_path(results_folder: str, filename: str, method: str) -> str:
    prefix = _TRANSFERABILITY_CSV_PREFIX.get((filename, method))
    if prefix is None:
        prefix = f"{filename}_{method}".replace(" ", "_")
    return os.path.join(results_folder, f"{prefix}.csv")


def _load_recorded_transfer_keys(csv_path: str) -> Set[Tuple[str, str, str]]:
    """
    Keys already present in the incremental CSV: (source_model, target_model, attack_name),
    with model columns equal to checkpoint stems (basename without ``.pt``).
    """
    recorded: Set[Tuple[str, str, str]] = set()
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return recorded
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return recorded
            for row in reader:
                try:
                    s = row.get("source_model", "").strip()
                    t = row.get("target_model", "").strip()
                    a = row.get("attack_name", "").strip()
                    if s and t and a:
                        recorded.add((s, t, a))
                except (TypeError, AttributeError):
                    continue
    except OSError:
        pass
    return recorded


class TransferabilityResult:
    """Class to store transferability attack results"""
    
    def __init__(self, source_model: str, target_model: str, attack_name: str, 
                 transfer_success: int, total_successful_attacks: int, total_images: int):
        self.source_model = source_model
        self.target_model = target_model
        self.attack_name = attack_name
        self.transfer_success = transfer_success
        self.total_successful_attacks = total_successful_attacks
        self.total_images = total_images
        self.transfer_rate = transfer_success / total_successful_attacks if total_successful_attacks > 0 else 0.0
        self.attack_success_rate = total_successful_attacks / total_images if total_images > 0 else 0.0


class TransferabilityLogger:
    """Class to handle logging and CSV saving for transferability experiments"""
    
    def __init__(self, results_folder: str = "results/transferability"):
        self.results_folder = results_folder
        self.failure_logs_folder = os.path.join(results_folder, "failure_logs")
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.failure_logs_folder, exist_ok=True)
        self._csv_path: Optional[str] = None
        self._recorded_keys: Set[Tuple[str, str, str]] = set()
    
    def begin_incremental_csv(self, filename: str, method: str) -> str:
        """
        Set the CSV path for this run. Creates the file with a header if it
        does not exist or is empty (fixed basename per experiment type).
        """
        self._csv_path = _transferability_csv_path(self.results_folder, filename, method)
        if (not os.path.isfile(self._csv_path)) or os.path.getsize(self._csv_path) == 0:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRANSFERABILITY_CSV_FIELDNAMES)
                w.writeheader()
            self._recorded_keys = set()
        else:
            self._recorded_keys = _load_recorded_transfer_keys(self._csv_path)
        print(f"📄 Results CSV: {self._csv_path}")
        return self._csv_path
    
    def has_recorded(self, source_model: str, target_model: str, attack_name: str) -> bool:
        """True if the results CSV already has a row for this source, target, and attack."""
        return (source_model, target_model, attack_name) in self._recorded_keys
    
    def print_duplicate_row_skip(
        self, source_model: str, target_model: str, attack_name: str
    ) -> None:
        """Console only: explain why a (source, target, attack) row is not written again."""
        print(
            f"⏭️ Skip (already in results CSV): {source_model} → "
            f"{target_model} ({attack_name})"
        )
    
    def append_result(self, result: TransferabilityResult) -> None:
        """Open CSV, append one row (with row timestamp), close."""
        if not self._csv_path:
            raise RuntimeError("append_result called before begin_incremental_csv")
        key = (result.source_model, result.target_model, result.attack_name)
        if key in self._recorded_keys:
            self.print_duplicate_row_skip(
                result.source_model, result.target_model, result.attack_name
            )
            return
        row_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "source_model": result.source_model,
            "target_model": result.target_model,
            "attack_name": result.attack_name,
            "total_images": result.total_images,
            "total_successful_attacks": result.total_successful_attacks,
            "transfer_success": result.transfer_success,
            "transfer_rate": result.transfer_rate,
            "attack_success_rate": result.attack_success_rate,
            "timestamp": row_ts,
        }
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TRANSFERABILITY_CSV_FIELDNAMES).writerow(row)
        self._recorded_keys.add(key)
    
    def save_failure_log(self, source_model: str, target_model: str, attack_name: str, 
                        exception: Exception, method: str):
        """Save failure log to file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failure_log_{source_model}_{target_model}_{attack_name}_{method}_{timestamp}.txt"
        filepath = os.path.join(self.failure_logs_folder, filename)
        
        with open(filepath, 'w') as f:
            f.write("TRANSFERABILITY FAILURE LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Source Model: {source_model}\n")
            f.write(f"Target Model: {target_model}\n")
            f.write(f"Attack: {attack_name}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Exception Type: {type(exception).__name__}\n")
            f.write(f"Exception Message: {str(exception)}\n\n")
            f.write("Full Traceback:\n")
            f.write("=" * 50 + "\n")
            f.write(traceback.format_exc())
    
    def save_results_to_csv(self, results: List[TransferabilityResult], 
                           filename: str, method: str):
        """Write all results in one shot (each row includes its own timestamp)."""
        csv_path = _transferability_csv_path(self.results_folder, filename, method)
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=TRANSFERABILITY_CSV_FIELDNAMES)
            writer.writeheader()
            for result in results:
                row_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow(
                    {
                        "source_model": result.source_model,
                        "target_model": result.target_model,
                        "attack_name": result.attack_name,
                        "total_images": result.total_images,
                        "total_successful_attacks": result.total_successful_attacks,
                        "transfer_success": result.transfer_success,
                        "transfer_rate": result.transfer_rate,
                        "attack_success_rate": result.attack_success_rate,
                        "timestamp": row_ts,
                    }
                )
        print(f"✅ Results saved to: {csv_path}")


def imagenette_transferability_model2model_in_memory(
    models: List[Tuple[str, str]],
    attack_names: List[str],
    images_per_attack: int = 100,
    batch_size: int = 1,
    results_folder: str = "results/transferability",
) -> List[TransferabilityResult]:
    """
    In-memory transferability: model to model.

    Args:
        models: List of ``(model_name, model_path)``. CSV ``source_model`` / ``target_model`` are
            checkpoint stems (basename without ``.pt``); ``model_name`` selects architecture when it is a known
            key and names saved-adversarial folders in from-files mode; ``model_path`` is the ``.pt`` file.
            If ``model_name`` is a known :class:`ModelNames` value, it selects the architecture;
            otherwise the architecture is inferred from ``model_path`` (see
            :func:`normalize_model_tuples`).
        attack_names: Attack algorithm names.
        images_per_attack: Max successful adversarial examples per source/attack.
        batch_size: DataLoader batch size.
        results_folder: Where incremental CSV and failure logs are written.

    Returns:
        List of TransferabilityResult objects
    """
    print("🔄 Starting in-memory model-to-model transferability analysis...")

    refs = normalize_model_tuples(models)

    logger = TransferabilityLogger(results_folder)
    results = []
    logger.begin_incremental_csv("model2model_transferability", "in_memory")

    _, test_loader = load_imagenette(batch_size=batch_size, test_subset_size=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for source in tqdm(refs, desc="M2M[in-mem] source models", unit="model"):
        source_label = source.label
        try:
            source_result = load_model_imagenette(
                source.path, source.architecture, device=device
            )
            if not source_result.success or source_result.model is None:
                err = source_result.error or "unknown"
                print(f"❌ Failed to load source {source_label}: {err}")
                logger.save_failure_log(
                    _checkpoint_csv_name(source.path), "N/A", "N/A", RuntimeError(err), "model2model"
                )
                continue
            source_model = source_result.model
            source_model.eval()

            for attack_name in tqdm(
                attack_names,
                desc=f"M2M[in-mem] {source_label} | attacks",
                leave=False,
                unit="attack",
            ):
                try:
                    pending_targets: List[ModelCheckpoint] = []
                    for t in refs:
                        if t.path == source.path:
                            continue
                        if logger.has_recorded(
                            _checkpoint_csv_name(source.path),
                            _checkpoint_csv_name(t.path),
                            attack_name,
                        ):
                            logger.print_duplicate_row_skip(
                                _checkpoint_csv_name(source.path),
                                _checkpoint_csv_name(t.path),
                                attack_name,
                            )
                        else:
                            pending_targets.append(t)
                    if not pending_targets:
                        continue
                    attack = AttackFactory.get_attack(attack_name, source_model)

                    adversarial_examples = []
                    source_labels = []
                    successful_attacks_count = 0

                    for images, labels in tqdm(
                        test_loader,
                        desc=f"M2M craft | src={source_label} atk={attack_name}",
                        leave=False,
                        unit="batch",
                    ):
                        if successful_attacks_count >= images_per_attack:
                            break

                        images, labels = images.to(device), labels.to(device)

                        images, labels = ModelUtils.remove_missclassified_imagenette(
                            source_model, images, labels
                        )

                        if labels.numel() == 0:
                            continue

                        adv_images = attack(images, labels)

                        with torch.no_grad():
                            source_outputs = source_model(adv_images)
                            source_predictions = torch.argmax(source_outputs, dim=1)

                            for i in range(len(adv_images)):
                                if successful_attacks_count >= images_per_attack:
                                    break

                                label = labels[i].item()
                                predicted_label = source_predictions[i].item()

                                if predicted_label != label:
                                    adversarial_examples.append(adv_images[i].cpu())
                                    source_labels.append(label)
                                    successful_attacks_count += 1

                    if successful_attacks_count == 0:
                        print(
                            f"⚠️ No successful attacks for {source_label} with {attack_name}"
                        )
                        continue

                    if adversarial_examples:
                        all_adv_images = torch.stack(adversarial_examples)
                        all_source_labels = torch.tensor(source_labels)
                        adv_gpu = all_adv_images.to(device)
                        labels_gpu = all_source_labels.to(device)
                        tb = _TARGET_TRANSFER_MICRO_BATCH

                        for target in tqdm(
                            pending_targets,
                            desc=f"M2M transfer | src={source_label} atk={attack_name}",
                            leave=False,
                            unit="target",
                        ):
                            target_model = None
                            target_result = None
                            try:
                                target_result = load_model_imagenette(
                                    target.path, target.architecture, device=device
                                )
                                if not target_result.success or target_result.model is None:
                                    print(
                                        f"❌ Failed to load target {target.label}: "
                                        f"{(target_result.error or 'unknown')}"
                                    )
                                    continue
                                target_model = target_result.model
                                target_model.eval()

                                transfer_success_count = 0
                                with torch.no_grad():
                                    for start in range(0, adv_gpu.size(0), tb):
                                        batch_adv = adv_gpu[start : start + tb]
                                        batch_lbl = labels_gpu[start : start + tb]
                                        target_outputs = target_model(batch_adv)
                                        target_predictions = torch.argmax(
                                            target_outputs, dim=1
                                        )
                                        transfer_success_count += (
                                            target_predictions != batch_lbl
                                        ).sum().item()

                                result = TransferabilityResult(
                                    source_model=_checkpoint_csv_name(source.path),
                                    target_model=_checkpoint_csv_name(target.path),
                                    attack_name=attack_name,
                                    transfer_success=transfer_success_count,
                                    total_successful_attacks=len(all_adv_images),
                                    total_images=successful_attacks_count,
                                )

                                results.append(result)
                                logger.append_result(result)

                                print(
                                    f"✅ {source_label} → {target.label} ({attack_name}): "
                                    f"{transfer_success_count}/{len(all_adv_images)} "
                                    f"({result.transfer_rate:.2%})"
                                )

                            except Exception as e:
                                print(
                                    f"❌ Error testing {source_label} → {target.label} ({attack_name}): {e}"
                                )
                                if _is_allocation_error(e):
                                    print(
                                        "↩️ CUDA allocation error — clearing cache and "
                                        "propagating for outer retry."
                                    )
                                    raise
                                logger.save_failure_log(
                                    _checkpoint_csv_name(source.path),
                                    _checkpoint_csv_name(target.path),
                                    attack_name,
                                    e,
                                    "model2model",
                                )
                            finally:
                                if target_model is not None:
                                    del target_model
                                if target_result is not None:
                                    del target_result
                                _defragment_cuda_memory()

                        del adv_gpu, labels_gpu
                        _defragment_cuda_memory()

                except Exception as e:
                    if _is_allocation_error(e):
                        raise
                    print(f"❌ Error with attack {attack_name} on {source_label}: {e}")
                    logger.save_failure_log(
                        _checkpoint_csv_name(source.path), "N/A", attack_name, e, "model2model"
                    )
                    continue

        except Exception as e:
            if _is_allocation_error(e):
                raise
            print(f"❌ Error loading source model {source_label}: {e}")
            logger.save_failure_log(
                _checkpoint_csv_name(source.path), "N/A", "N/A", e, "model2model"
            )
            continue

    return results


def imagenette_transferability_attack2model_in_memory(
    models: List[Tuple[str, str]],
    attack_names: List[str],
    images_per_attack: int = 100,
    batch_size: int = 1,
    results_folder: str = "results/transferability",
) -> List[TransferabilityResult]:
    """
    In-memory attack-to-model transferability.

    Adversarial examples are crafted on the **first** ``(model_name, model_path)`` pair; transfer
    is evaluated on the remaining checkpoints. See :func:`imagenette_transferability_model2model_in_memory`
    for tuple semantics.
    """
    print("🔄 Starting in-memory attack-to-model transferability analysis...")

    refs = normalize_model_tuples(models)
    if len(refs) < 2:
        raise ValueError("attack2model transferability requires at least two (model_name, model_path) entries")

    source = refs[0]

    logger = TransferabilityLogger(results_folder)
    results = []
    logger.begin_incremental_csv("attack2model_transferability", "in_memory")
    _, test_loader = load_imagenette(batch_size=batch_size, test_subset_size=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for attack_name in tqdm(
        attack_names,
        desc=f"A2M[in-mem] src={source.label} | attacks",
        unit="attack",
    ):
        try:
            pending_targets: List[ModelCheckpoint] = []
            for t in refs:
                if t.path == source.path:
                    continue
                if logger.has_recorded(
                    _checkpoint_csv_name(source.path),
                    _checkpoint_csv_name(t.path),
                    attack_name,
                ):
                    logger.print_duplicate_row_skip(
                        _checkpoint_csv_name(source.path),
                        _checkpoint_csv_name(t.path),
                        attack_name,
                    )
                else:
                    pending_targets.append(t)
            if not pending_targets:
                continue

            source_result = load_model_imagenette(
                source.path, source.architecture, device=device
            )
            if not source_result.success or source_result.model is None:
                err = source_result.error or "unknown"
                print(f"❌ Failed to load source {source.label}: {err}")
                logger.save_failure_log(
                    _checkpoint_csv_name(source.path), "N/A", attack_name, RuntimeError(err), "attack2model"
                )
                continue
            source_model = source_result.model
            source_model.eval()

            attack = AttackFactory.get_attack(attack_name, source_model)

            adversarial_examples = []
            source_labels = []
            successful_attacks_count = 0

            for images, labels in tqdm(
                test_loader,
                desc=f"A2M craft | src={source.label} atk={attack_name}",
                leave=False,
                unit="batch",
            ):
                if successful_attacks_count >= images_per_attack:
                    break

                images, labels = images.to(device), labels.to(device)

                images, labels = ModelUtils.remove_missclassified_imagenette(
                    source_model, images, labels
                )

                if labels.numel() == 0:
                    continue

                adv_images = attack(images, labels)

                with torch.no_grad():
                    source_outputs = source_model(adv_images)
                    source_predictions = torch.argmax(source_outputs, dim=1)

                    for i in range(len(adv_images)):
                        if successful_attacks_count >= images_per_attack:
                            break

                        label = labels[i].item()
                        predicted_label = source_predictions[i].item()

                        if predicted_label != label:
                            adversarial_examples.append(adv_images[i].cpu())
                            source_labels.append(label)
                            successful_attacks_count += 1

            if successful_attacks_count == 0:
                print(f"⚠️ No successful attacks for {attack_name}")
                continue

            if adversarial_examples:
                all_adv_images = torch.stack(adversarial_examples)
                all_source_labels = torch.tensor(source_labels)
                adv_gpu = all_adv_images.to(device)
                labels_gpu = all_source_labels.to(device)
                tb = _TARGET_TRANSFER_MICRO_BATCH

                for target in tqdm(
                    pending_targets,
                    desc=f"A2M transfer | src={source.label} atk={attack_name}",
                    leave=False,
                    unit="target",
                ):
                    target_model = None
                    target_result = None
                    try:
                        target_result = load_model_imagenette(
                            target.path, target.architecture, device=device
                        )
                        if not target_result.success or target_result.model is None:
                            print(
                                f"❌ Failed to load target {target.label}: "
                                f"{(target_result.error or 'unknown')}"
                            )
                            continue
                        target_model = target_result.model
                        target_model.eval()

                        transfer_success_count = 0
                        with torch.no_grad():
                            for start in range(0, adv_gpu.size(0), tb):
                                batch_adv = adv_gpu[start : start + tb]
                                batch_lbl = labels_gpu[start : start + tb]
                                target_outputs = target_model(batch_adv)
                                target_predictions = torch.argmax(target_outputs, dim=1)
                                transfer_success_count += (
                                    target_predictions != batch_lbl
                                ).sum().item()

                        result = TransferabilityResult(
                            source_model=_checkpoint_csv_name(source.path),
                            target_model=_checkpoint_csv_name(target.path),
                            attack_name=attack_name,
                            transfer_success=transfer_success_count,
                            total_successful_attacks=len(all_adv_images),
                            total_images=successful_attacks_count,
                        )

                        results.append(result)
                        logger.append_result(result)

                        print(
                            f"✅ {attack_name} → {target.label}: "
                            f"{transfer_success_count}/{len(all_adv_images)} "
                            f"({result.transfer_rate:.2%})"
                        )

                    except Exception as e:
                        print(f"❌ Error testing {attack_name} → {target.label}: {e}")
                        if _is_allocation_error(e):
                            print(
                                "↩️ CUDA allocation error — clearing cache and "
                                "propagating for outer retry."
                            )
                            raise
                        logger.save_failure_log(
                            _checkpoint_csv_name(source.path),
                            _checkpoint_csv_name(target.path),
                            attack_name,
                            e,
                            "attack2model",
                        )
                    finally:
                        if target_model is not None:
                            del target_model
                        if target_result is not None:
                            del target_result
                        _defragment_cuda_memory()

                del adv_gpu, labels_gpu
                _defragment_cuda_memory()

        except Exception as e:
            if _is_allocation_error(e):
                raise
            print(f"❌ Error with attack {attack_name}: {e}")
            logger.save_failure_log("N/A", "N/A", attack_name, e, "attack2model")
            continue

    return results


def imagenette_transferability_model2model_from_files(
    models: List[Tuple[str, str]],
    attack_names: List[str],
    attacked_images_folder: str = "data/attacks/imagenette_models",
    results_folder: str = "results/transferability",
) -> List[TransferabilityResult]:
    """
    File-based model-to-model transferability.

    Adversarial PNGs are read from folders named by each tuple's ``model_name`` (CSV label):
    ``{attacked_images_folder}/{model_name}/{attack}/...`` or the ``train``/``test`` layouts
    handled by :func:`_resolve_model_adv_images_dir`. Checkpoints are loaded from each
    ``model_path`` (see :func:`normalize_model_tuples`).
    """
    print("🔄 Starting file-based model-to-model transferability analysis...")

    refs = normalize_model_tuples(models)

    logger = TransferabilityLogger(results_folder)
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.begin_incremental_csv("model2model_transferability", "from_files")

    for source in tqdm(refs, desc="M2M[files] source models", unit="model"):
        source_label = source.label
        for attack_name in tqdm(
            attack_names,
            desc=f"M2M[files] {source_label} | attacks",
            leave=False,
            unit="attack",
        ):
            try:
                pending_targets: List[ModelCheckpoint] = []
                for t in refs:
                    if t.path == source.path:
                        continue
                    if logger.has_recorded(
                        _checkpoint_csv_name(source.path),
                        _checkpoint_csv_name(t.path),
                        attack_name,
                    ):
                        logger.print_duplicate_row_skip(
                            _checkpoint_csv_name(source.path),
                            _checkpoint_csv_name(t.path),
                            attack_name,
                        )
                    else:
                        pending_targets.append(t)
                if not pending_targets:
                    continue

                adv_images_path = _resolve_model_adv_images_dir(
                    attacked_images_folder, source_label, attack_name
                )

                if adv_images_path is None:
                    print(
                        f"⚠️ No saved images found for {source_label}/{attack_name} "
                        f"(under {attacked_images_folder})"
                    )
                    continue

                adversarial_examples = []
                source_labels = []

                class_folders = [
                    d
                    for d in os.listdir(adv_images_path)
                    if os.path.isdir(os.path.join(adv_images_path, d))
                ]

                for class_folder in class_folders:
                    class_path = os.path.join(adv_images_path, class_folder)
                    class_label = int(class_folder)

                    image_files = [
                        f
                        for f in os.listdir(class_path)
                        if f.endswith(".png") and not f.startswith("src_")
                    ]

                    for image_file in image_files:
                        image_path = os.path.join(class_path, image_file)
                        try:
                            image = Image.open(image_path).convert("RGB")
                            image_tensor = (
                                torch.from_numpy(np.array(image))
                                .permute(2, 0, 1)
                                .float()
                                / 255.0
                            )
                            adversarial_examples.append(image_tensor)
                            source_labels.append(class_label)
                        except Exception as e:
                            print(f"⚠️ Error loading {image_path}: {e}")
                            continue

                if not adversarial_examples:
                    print(
                        f"⚠️ No valid adversarial images found for {source_label}/{attack_name}"
                    )
                    continue

                all_adv_images = torch.stack(adversarial_examples)
                all_source_labels = torch.tensor(source_labels)
                successful_attacks_count = len(adversarial_examples)

                print(
                    f"📁 Loaded {successful_attacks_count} adversarial images for "
                    f"{source_label}/{attack_name}"
                )

                for target in pending_targets:
                    try:
                        target_result = load_model_imagenette(
                            target.path, target.architecture, device=device
                        )
                        if not target_result.success:
                            print(
                                f"❌ Failed to load target {target.label}: "
                                f"{(target_result.error or 'unknown')}"
                            )
                            continue
                        target_model = target_result.model
                        target_model.eval()

                        transfer_success_count = 0
                        batch_sz = 32
                        for i in range(0, len(all_adv_images), batch_sz):
                            batch_adv = all_adv_images[i : i + batch_sz].to(device)
                            batch_labels = all_source_labels[i : i + batch_sz].to(device)

                            with torch.no_grad():
                                target_outputs = target_model(batch_adv)
                                target_predictions = torch.argmax(target_outputs, dim=1)

                                transfer_mask = target_predictions != batch_labels
                                transfer_success_count += transfer_mask.sum().item()

                        result = TransferabilityResult(
                            source_model=_checkpoint_csv_name(source.path),
                            target_model=_checkpoint_csv_name(target.path),
                            attack_name=attack_name,
                            transfer_success=transfer_success_count,
                            total_successful_attacks=len(all_adv_images),
                            total_images=len(all_adv_images),
                        )

                        results.append(result)
                        logger.append_result(result)

                        print(
                            f"✅ {source_label} → {target.label} ({attack_name}): "
                            f"{transfer_success_count}/{len(all_adv_images)} "
                            f"({result.transfer_rate:.2%})"
                        )

                    except Exception as e:
                        print(
                            f"❌ Error testing {source_label} → {target.label} ({attack_name}): {e}"
                        )
                        logger.save_failure_log(
                            _checkpoint_csv_name(source.path),
                            _checkpoint_csv_name(target.path),
                            attack_name,
                            e,
                            "model2model_files",
                        )
                        continue

            except Exception as e:
                print(f"❌ Error processing {source_label}/{attack_name}: {e}")
                logger.save_failure_log(
                    _checkpoint_csv_name(source.path), "N/A", attack_name, e, "model2model_files"
                )
                continue

    return results


def imagenette_transferability_attack2model_from_files(
    models: List[Tuple[str, str]],
    attack_names: List[str],
    attacked_images_folder: str = "data/attacks/imagenette_models",
    results_folder: str = "results/transferability",
) -> List[TransferabilityResult]:
    """
    File-based attack-to-model transferability.

    Uses the **first** ``(model_name, model_path)`` pair for saved adversarial images (folder name =
    ``model_name``). Targets are the remaining checkpoints.
    """
    print("🔄 Starting file-based attack-to-model transferability analysis...")

    refs = normalize_model_tuples(models)
    if len(refs) < 2:
        raise ValueError("At least two (model_name, model_path) entries are required")

    source = refs[0]

    logger = TransferabilityLogger(results_folder)
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.begin_incremental_csv("attack2model_transferability", "from_files")

    for attack_name in tqdm(
        attack_names,
        desc=f"A2M[files] src={source.label} | attacks",
        unit="attack",
    ):
        try:
            pending_targets: List[ModelCheckpoint] = []
            for t in refs:
                if t.path == source.path:
                    continue
                if logger.has_recorded(
                    _checkpoint_csv_name(source.path),
                    _checkpoint_csv_name(t.path),
                    attack_name,
                ):
                    logger.print_duplicate_row_skip(
                        _checkpoint_csv_name(source.path),
                        _checkpoint_csv_name(t.path),
                        attack_name,
                    )
                else:
                    pending_targets.append(t)
            if not pending_targets:
                continue

            adv_images_path = _resolve_saved_adv_images_dir(
                attacked_images_folder, source.label, attack_name
            )
            if adv_images_path is None:
                print(
                    f"⚠️ No saved images found for {source.label}/{attack_name} "
                    f"(tried train/, test/, and flat layout under {attacked_images_folder})"
                )
                continue

            adversarial_examples = []
            source_labels = []

            class_folders = [
                d
                for d in os.listdir(adv_images_path)
                if os.path.isdir(os.path.join(adv_images_path, d))
            ]

            for class_folder in class_folders:
                class_path = os.path.join(adv_images_path, class_folder)
                class_label = int(class_folder)

                image_files = [
                    f
                    for f in os.listdir(class_path)
                    if f.endswith(".png") and not f.startswith("src_")
                ]

                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    try:
                        image = Image.open(image_path).convert("RGB")
                        image_tensor = (
                            torch.from_numpy(np.array(image))
                            .permute(2, 0, 1)
                            .float()
                            / 255.0
                        )
                        adversarial_examples.append(image_tensor)
                        source_labels.append(class_label)
                    except Exception as e:
                        print(f"⚠️ Error loading {image_path}: {e}")
                        continue

            if not adversarial_examples:
                print(f"⚠️ No valid adversarial images found for {attack_name}")
                continue

            all_adv_images = torch.stack(adversarial_examples)
            all_source_labels = torch.tensor(source_labels)
            successful_attacks_count = len(adversarial_examples)

            print(
                f"📁 Loaded {successful_attacks_count} adversarial images for "
                f"{source.label} / {attack_name}"
            )

            for target in tqdm(
                pending_targets,
                desc=f"A2M[files] transfer | src={source.label} atk={attack_name}",
                leave=False,
                unit="target",
            ):
                try:
                    target_result = load_model_imagenette(
                        target.path, target.architecture, device=device
                    )
                    if not target_result.success:
                        print(
                            f"❌ Failed to load target {target.label}: "
                            f"{(target_result.error or 'unknown')}"
                        )
                        continue
                    target_model = target_result.model
                    target_model.eval()

                    transfer_success_count = 0
                    batch_sz = 32
                    for i in range(0, len(all_adv_images), batch_sz):
                        batch_adv = all_adv_images[i : i + batch_sz].to(device)
                        batch_labels = all_source_labels[i : i + batch_sz].to(device)

                        with torch.no_grad():
                            target_outputs = target_model(batch_adv)
                            target_predictions = torch.argmax(target_outputs, dim=1)

                            transfer_mask = target_predictions != batch_labels
                            transfer_success_count += transfer_mask.sum().item()

                    result = TransferabilityResult(
                        source_model=_checkpoint_csv_name(source.path),
                        target_model=_checkpoint_csv_name(target.path),
                        attack_name=attack_name,
                        transfer_success=transfer_success_count,
                        total_successful_attacks=len(all_adv_images),
                        total_images=successful_attacks_count,
                    )

                    results.append(result)
                    logger.append_result(result)

                    print(
                        f"✅ {attack_name} → {target.label}: "
                        f"{transfer_success_count}/{len(all_adv_images)} "
                        f"({result.transfer_rate:.2%})"
                    )

                except Exception as e:
                    print(f"❌ Error testing {attack_name} → {target.label}: {e}")
                    logger.save_failure_log(
                        _checkpoint_csv_name(source.path),
                        _checkpoint_csv_name(target.path),
                        attack_name,
                        e,
                        "attack2model_files",
                    )
                    continue

        except Exception as e:
            print(f"❌ Error with attack {attack_name}: {e}")
            logger.save_failure_log("N/A", "N/A", attack_name, e, "attack2model_files")
            continue

    return results


def run_all_transferability_experiments(
    models: Optional[List[Tuple[str, str]]] = None,
    attack_names: Optional[List[str]] = None,
    images_per_attack: int = 100,
    attacked_images_folder: str = "data/attacks/imagenette_models",
    results_folder: str = "results/transferability",
    models_root: str = "./models/imagenette",
    allocation_retry_min_elapsed_sec: float = 60.0,
    allocation_retry_max_attempts: int = 50,
):
    """
    Run all transferability experiments.

    Args:
        models: List of ``(model_name, model_path)``. If omitted, uses a default set of
            ``(architecture, {models_root}/{architecture}_advanced.pt)`` pairs.
        attack_names: Optional; defaults to a small FGSM/PGD/... set.
        models_root: Used only when ``models`` is None to build default checkpoint paths.
        allocation_retry_min_elapsed_sec: See :func:`run_transferability_with_allocation_retry`.
        allocation_retry_max_attempts: See :func:`run_transferability_with_allocation_retry`.
    """
    print("🚀 Starting comprehensive transferability analysis...")

    if models is None:
        names = [
            ModelNames().resnet18,
            ModelNames().densenet121,
            ModelNames().mobilenet_v2,
            ModelNames().efficientnet_b0,
        ]
        models = [
            (n, os.path.normpath(os.path.join(models_root, f"{n}_advanced.pt"))) for n in names
        ]

    if attack_names is None:
        attack_names = [
            AttackNames().FGSM,
            AttackNames().PGD,
            AttackNames().BIM,
            AttackNames().FFGSM,
            AttackNames().TPGD,
        ]

    print(f"📊 Models: {len(models)} checkpoint(s)")
    print(f"🎯 Attacks: {attack_names}")

    experiments = [
        (
            "In-Memory Model-to-Model",
            lambda: imagenette_transferability_model2model_in_memory(
                models=models,
                attack_names=attack_names,
                images_per_attack=images_per_attack,
                results_folder=results_folder,
            ),
        ),
        (
            "In-Memory Attack-to-Model",
            lambda: imagenette_transferability_attack2model_in_memory(
                models=models,
                attack_names=attack_names,
                images_per_attack=images_per_attack,
                results_folder=results_folder,
            ),
        ),
        (
            "File-Based Model-to-Model",
            lambda: imagenette_transferability_model2model_from_files(
                models=models,
                attack_names=attack_names,
                attacked_images_folder=attacked_images_folder,
                results_folder=results_folder,
            ),
        ),
        (
            "File-Based Attack-to-Model",
            lambda: imagenette_transferability_attack2model_from_files(
                models=models,
                attack_names=attack_names,
                attacked_images_folder=attacked_images_folder,
                results_folder=results_folder,
            ),
        ),
    ]
    
    all_results = {}
    
    for exp_name, exp_func in experiments:
        print(f"\n{'='*60}")
        print(f"🔄 Running: {exp_name}")
        print(f"{'='*60}")
        
        try:
            results = run_transferability_with_allocation_retry(
                exp_func,
                min_elapsed_seconds_to_retry=allocation_retry_min_elapsed_sec,
                max_allocation_retries=allocation_retry_max_attempts,
            )
            all_results[exp_name] = results
            print(f"✅ {exp_name} completed successfully!")
            
        except Exception as e:
            print(f"❌ {exp_name} failed: {e}")
            continue
    
    print(f"\n🎉 All transferability experiments completed!")
    print(f"📁 Results saved in: {results_folder}")
    
    return all_results


if __name__ == "__main__":
    attack_names = AttackNames().all_attack_names
    save_path = "results/imagenette/transferability/model2model"

    # (model_name, model_path): model_name = label in CSV / adversarial folders; path = .pt file.
    # If model_name is a known architecture (e.g. resnet18), it selects the torchvision model.
    # Otherwise the architecture is inferred from the filename.
    models: List[Tuple[str, str]] = [
        (ModelNames().densenet121, "./models/imagenette_adversarial/densenet121_adv_preattacked_20260415.pt"),
        (ModelNames().efficientnet_b0, "./models/imagenette_adversarial/efficientnet_b0_adv_preattacked_20260415.pt"),
        (ModelNames().mobilenet_v2, "./models/imagenette_adversarial/mobilenet_v2_adv_preattacked_20260415.pt"),
        (ModelNames().resnet18, "./models/imagenette_adversarial/resnet18_adv_preattacked_20260415.pt"),
        (ModelNames().vgg16, "./models/imagenette_adversarial/vgg16_adv_preattacked_20260415.pt"),
        (ModelNames().resnet18, "./models/imagenette/resnet18_advanced.pt"),
        (ModelNames().densenet121, "./models/imagenette/densenet121_advanced.pt"),
        (ModelNames().mobilenet_v2, "./models/imagenette/mobilenet_v2_advanced.pt"),
        (ModelNames().efficientnet_b0, "./models/imagenette/efficientnet_b0_advanced.pt"),
        (ModelNames().vgg16, "./models/imagenette/vgg16_advanced.pt"),
    ]

    run_transferability_with_allocation_retry(
        lambda: imagenette_transferability_model2model_in_memory(
            models=models,
            attack_names=attack_names,
            images_per_attack=500,
            results_folder=save_path,
        ),
    )