import gc
import math
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import random
import logging
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import hashlib

from av.video.reformatter import VideoReformatter
from scipy.fftpack import dct as scipy_dct, idct as scipy_idct
from scipy.linalg import svd as scipy_svd

from galois import BCH
# from PIL import Image
from line_profiler import line_profiler, profile

import torch
import torch.nn.functional as F
try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    class DTCWTForward: pass
    class DTCWTInverse: pass
    logging.error("Библиотека pytorch_wavelets не найдена!")
try:
    import torch_dct as dct_torch
    TORCH_DCT_AVAILABLE = True
except ImportError:
    TORCH_DCT_AVAILABLE = False
    logging.error("Библиотека torch-dct не найдена! Невозможно использовать PyTorch DCT.")

from typing import List, Tuple, Optional, Dict, Any, Set
import uuid
from math import ceil
import cProfile
import pstats
from fractions import Fraction

PYMEDIAINFO_AVAILABLE = False
try:
    from pymediainfo import MediaInfo
    PYMEDIAINFO_AVAILABLE = True
    logging.info("pymediainfo library imported successfully.")
except ImportError:
    logging.warning("pymediainfo library not found. Install it: pip install pymediainfo. MediaInfo fallback will not be available.")
    class MediaInfo:
        def __init__(self, xml): pass
        @staticmethod
        def can_parse(): return False
        def Tofile(self, filepath): return self
        @property
        def tracks(self): return []

try:
    import av
    from av import FFmpegError, VideoFrame
    from av import EOFError as FFmpegEOFError
    from av import ValueError as FFmpegValueError

    PYAV_AVAILABLE = True
    logging.info("PyAV library imported successfully.")

except ImportError:
    PYAV_AVAILABLE = False
    logging.error("PyAV library not found! Install it: pip install av")
    class av_dummy:
        class VideoFrame: pass
        class AudioFrame: pass
        class Packet: pass
        class TimeBase: pass
        class container:
            class Container: pass
        FFmpegError = Exception
        EOFError = EOFError
        ValueError = ValueError
        NotFoundError = Exception
    av = av_dummy
    FFmpegError = Exception
    FFmpegEOFError = EOFError
    FFmpegValueError = ValueError

try:
    import galois
    BCH_TYPE = galois.BCH; GALOIS_IMPORTED = True; logging.info("galois library imported.")
except ImportError:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.info("galois library not found.")
except Exception as import_err:
    class BCH: pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.error(f"Galois import error: {import_err}", exc_info=True)

CODEC_CONTAINER_COMPATIBILITY: Dict[str, Set[Tuple[str, str]]] = {
    ".mp4": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('audio', 'aac'), ('audio', 'mp3'), ('audio', 'alac')},
    ".mov": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('video', 'prores'), ('audio', 'aac'), ('audio', 'mp3'), ('audio', 'alac'), ('audio', 'pcm_s16le')},
    ".mkv": {('video', 'h264'), ('video', 'hevc'), ('video', 'vp9'), ('video', 'av1'), ('video', 'mpeg4'), ('video', 'mpeg2video'), ('video', 'theora'), ('video', 'prores'), ('audio', 'aac'), ('audio', 'opus'), ('audio', 'vorbis'), ('audio', 'flac'), ('audio', 'ac3'), ('audio', 'dts'), ('audio', 'mp3'), ('audio', 'pcm_s16le'), ('audio', 'alac')},
    ".webm": {('video', 'vp8'), ('video', 'vp9'), ('video', 'av1'), ('audio', 'opus'), ('audio', 'vorbis')},
}
DEFAULT_OUTPUT_CONTAINER_EXT_FINAL: str = ".mp4"
DEFAULT_VIDEO_ENCODER_LIB_FOR_HEAD: str = "libx264"
DEFAULT_VIDEO_CODEC_NAME_FOR_HEAD: str = "h264"
DEFAULT_AUDIO_CODEC_FOR_FFMPEG: str = "aac"
FALLBACK_CONTAINER_EXT_FINAL: str = ".mkv"


LAMBDA_PARAM: float = 0.05
ALPHA_MIN: float = 1.13
ALPHA_MAX: float = 1.28
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2 # Cb
USE_PERCEPTUAL_MASKING: bool = True
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection'
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 9
FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed_pytorch.log'
OUTPUT_CODEC: str = 'mp4v'
OUTPUT_EXTENSION: str = '.mp4'
SELECTED_RINGS_FILE: str = 'selected_rings_embed_pytorch.json'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS: Optional[int] = 8
MAX_TOTAL_PACKETS = 15
SAFE_MAX_WORKERS = 8

BCH_CODE_OBJECT: Optional['galois.BCH'] = None
GALOIS_AVAILABLE = False
try:
    import galois

    logging.info("galois: импортирован.")
    _test_bch_ok = False;
    _test_decode_ok = False
    try:
        _test_m = BCH_M
        _test_t = BCH_T
        _test_n = (1 << _test_m) - 1
        _test_d = 2 * _test_t + 1

        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)

        if _test_t == 5:
            expected_k = 215
        elif _test_t == 7:
            expected_k = 201
        elif _test_t == 9:
            expected_k = 187
        elif _test_t == 11:
            expected_k = 173
        elif _test_t == 15:
            expected_k = 131
        else:
            logging.error(f"Неизвестное ожидаемое k для t={_test_t}")
            expected_k = -1

        if expected_k != -1 and _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
            logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
            _test_bch_ok = True;
            BCH_CODE_OBJECT = _test_bch_galois
        else:
            logging.error(
                f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t={_test_bch_galois.t}, k={_test_bch_galois.k}.")
            _test_bch_ok = False;
            BCH_CODE_OBJECT = None

        if _test_bch_ok and BCH_CODE_OBJECT is not None:
            _n_bits = BCH_CODE_OBJECT.n
            _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)
            GF2 = galois.GF(2);
            _dummy_cw_vec = GF2(_dummy_cw_bits)
            _msg, _flips = BCH_CODE_OBJECT.decode(_dummy_cw_vec, errors=True)
            if _flips is not None:
                logging.info(f"galois: decode() test OK (flips={_flips}).")
                _test_decode_ok = True
            else:
                _test_decode_ok = True
                logging.info("galois: decode() test potentially OK (flips is None/0).")
    except ValueError as ve:
        logging.error(f"galois: ОШИБКА ValueError при инициализации BCH: {ve}")
        BCH_CODE_OBJECT = None;
        _test_bch_ok = False
    except Exception as test_err:
        logging.error(f"galois: ОШИБКА теста инициализации/декодирования: {test_err}", exc_info=True)
        BCH_CODE_OBJECT = None;
        _test_bch_ok = False

    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if GALOIS_AVAILABLE:
        logging.info("galois: Тесты пройдены, объект BCH доступен.")
    else:
        logging.warning("galois: Тесты НЕ ПРОЙДЕНЫ или объект BCH не создан.")

except ImportError:
    GALOIS_AVAILABLE = False;
    BCH_CODE_OBJECT = None
    logging.info("galois library not found.")
except Exception as import_err:
    GALOIS_AVAILABLE = False;
    BCH_CODE_OBJECT = None
    logging.error(f"galois: Ошибка импорта: {import_err}", exc_info=True)

# --- Настройка логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Встраивания (PyTorch Wavelets) ---")
logging.info(f"PyTorch Wavelets Доступно: {PYTORCH_WAVELETS_AVAILABLE}")
logging.info(f"Torch DCT Доступно: {TORCH_DCT_AVAILABLE}")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Payload: {PAYLOAD_LEN_BYTES * 8}bit, ECC for 1st: {effective_use_ecc} (Galois BCH m={BCH_M}, t={BCH_T})")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(
    f"Маскировка: {USE_PERCEPTUAL_MASKING} (Lambda={LAMBDA_PARAM}), Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS or 'default'}) с батчингом.")
if USE_ECC and not GALOIS_AVAILABLE:
    logging.warning("ECC вкл, но galois недоступна/не работает! Первый пакет будет Raw.")
elif not USE_ECC:
    logging.info("ECC выкл для первого пакета.")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG', 'DIVX']: logging.warning(f"Используется кодек с потерями '{OUTPUT_CODEC}'.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE:
    logging.error(f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) > CANDIDATE_POOL_SIZE ({CANDIDATE_POOL_SIZE})! Исправлено.")
    NUM_RINGS_TO_USE = CANDIDATE_POOL_SIZE
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(
    f"NUM_RINGS_TO_USE ({NUM_RINGS_TO_USE}) != BITS_PER_PAIR ({BITS_PER_PAIR}).")


# --- Базовые Функции ---

def dct1d_torch(s_tensor: torch.Tensor) -> torch.Tensor:
    """1D DCT-II используя torch-dct."""
    if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct не доступен")
    # dct ожидает тензор и применяет преобразование к последнему измерению
    return dct_torch.dct(s_tensor, norm='ortho')

def idct1d_torch(c_tensor: torch.Tensor) -> torch.Tensor:
    """1D IDCT-III используя torch-dct."""
    if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct не доступен")
    return dct_torch.idct(c_tensor, norm='ortho')

def svd_torch(tensor_1d: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Применяет SVD к 1D тензору (рассматривая как столбец) используя torch.linalg.svd."""
    try:
        tensor_2d = tensor_1d.unsqueeze(-1)
        U, S, Vh = torch.linalg.svd(tensor_2d, full_matrices=False)
        # S - вектор сингулярных чисел
        # U - матрица Nx1
        return U, S, Vh.T
    except Exception as e:
        logging.error(f"PyTorch SVD error: {e}", exc_info=True)
        return None, None, None

# ---  обертка для PyTorch DTCWT Forward ---
def dtcwt_pytorch_forward(yp_tensor: torch.Tensor, xfm: DTCWTForward, device: torch.device, fn: int = -1) -> Tuple[
    Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Применяет прямое DTCWT PyTorch к одному каналу (2D тензору)."""
    if not PYTORCH_WAVELETS_AVAILABLE:
        logging.error("PyTorch Wavelets не доступна для dtcwt_pytorch_forward.")
        return None, None
    if not isinstance(yp_tensor, torch.Tensor):
        logging.error(f"[Frame:{fn}] Input is not a torch Tensor.")
        return None, None
    if yp_tensor.ndim != 2:
        logging.error(f"[Frame:{fn}] Input tensor must be 2D (H, W), got {yp_tensor.shape}")
        return None, None
    try:
        yp_tensor = yp_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)  # -> (1, 1, H, W)
        xfm = xfm.to(device)

        with torch.no_grad():
            Yl, Yh = xfm(yp_tensor)

        # Проверка результата
        if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh:
            logging.error(f"[Frame:{fn}] DTCWTForward вернула некорректный результат (None или пустой Yh).")
            return None, None

        # logging.debug(f"[Frame:{fn}] PyTorch DTCWT FWD done. Yl shape: {Yl.shape}, Yh[0] shape: {Yh[0].shape}")
        return Yl, Yh
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error(f"[Frame:{fn}] CUDA out of memory during PyTorch DTCWT forward!", exc_info=True)
        else:
            logging.error(f"[Frame:{fn}] PyTorch DTCWT forward runtime error: {e}", exc_info=True)
        return None, None
    except Exception as e:
        logging.error(f"[Frame:{fn}] PyTorch DTCWT forward unexpected error: {e}", exc_info=True)
        return None, None


# --- функция обертка для PyTorch DTCWT Inverse ---
def dtcwt_pytorch_inverse(Yl: torch.Tensor, Yh: List[torch.Tensor], ifm: DTCWTInverse, device: torch.device,
                          target_shape: Tuple[int, int], fn: int = -1) -> Optional[np.ndarray]:
    """Применяет обратное DTCWT PyTorch и возвращает NumPy массив float32."""
    if not PYTORCH_WAVELETS_AVAILABLE:
        logging.error("PyTorch Wavelets не доступна для dtcwt_pytorch_inverse.")
        return None
    if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh:
        logging.error(f"[Frame:{fn}] Invalid input for inverse (Yl or Yh is None/empty list).")
        return None
    try:
        # Перемещаем все на device
        Yl = Yl.to(device)
        Yh = [h.to(device) for h in Yh if h is not None and h.numel() > 0]  # Фильтруем пустые/None
        ifm = ifm.to(device)

        if not Yh:
            logging.error(f"[Frame:{fn}] Yh list is empty after filtering Nones/empty tensors.")
            return None

        with torch.no_grad():
            reconstructed_X_tensor = ifm((Yl, Yh))

        if reconstructed_X_tensor.dim() == 4 and reconstructed_X_tensor.shape[0] == 1 and reconstructed_X_tensor.shape[
            1] == 1:
            reconstructed_X_tensor = reconstructed_X_tensor.squeeze(0).squeeze(0)  # (H, W)
        elif reconstructed_X_tensor.dim() != 2:
            logging.error(f"[Frame:{fn}] Unexpected output dimension from inverse: {reconstructed_X_tensor.dim()}")
            return None

        # logging.debug(f"[Frame:{fn}] PyTorch DTCWT INV done. Output shape: {reconstructed_X_tensor.shape}")

        current_h, current_w = reconstructed_X_tensor.shape
        target_h, target_w = target_shape
        if current_h > target_h or current_w > target_w:
            logging.warning(
                f"[Frame:{fn}] Inverse result shape {reconstructed_X_tensor.shape} > target {target_shape}. Cropping.")
            if target_h > 0 and target_w > 0:
                reconstructed_X_tensor = reconstructed_X_tensor[:target_h, :target_w]
            else:
                logging.error(f"[Frame:{fn}] Invalid target shape for cropping: {target_shape}")
                return None
        elif current_h < target_h or current_w < target_w:
            logging.warning(
                f"[Frame:{fn}] Inverse result shape {reconstructed_X_tensor.shape} < target {target_shape}. Padding might be needed if this causes issues.")

        # Перемещаем на CPU и конвертиртация в NumPy float32
        reconstructed_np = reconstructed_X_tensor.cpu().numpy().astype(np.float32)

        if np.any(np.isnan(reconstructed_np)):
            logging.warning(f"[Frame:{fn}] NaN found after PyTorch inverse!")

        return reconstructed_np
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error(f"[Frame:{fn}] CUDA out of memory during PyTorch DTCWT inverse!", exc_info=True)
        else:
            logging.error(f"[Frame:{fn}] PyTorch DTCWT inverse runtime error: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"[Frame:{fn}] PyTorch DTCWT inverse unexpected error: {e}", exc_info=True)
        return None


def ring_division(lp_tensor: torch.Tensor, nr: int = N_RINGS, fn: int = -1) -> List[Optional[torch.Tensor]]:
    """
    Разбивает 2D PyTorch тензор на N концентрических колец.
    Возвращает список тензоров координат (GPU-версия).
    """
    if not isinstance(lp_tensor, torch.Tensor) or lp_tensor.ndim != 2:
        logging.error(
            f"[Frame:{fn}] Invalid input for ring_division (expected 2D torch.Tensor). Got {type(lp_tensor)} with ndim {lp_tensor.ndim if hasattr(lp_tensor, 'ndim') else 'N/A'}")
        return [None] * nr

    H, W = lp_tensor.shape
    if H < 2 or W < 2:
        logging.warning(f"[Frame:{fn}] Tensor too small for ring division ({H}x{W})")
        return [None] * nr
    device = lp_tensor.device

    try:
        rr, cc = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32),
                                indexing='ij')

        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
        distances = torch.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)

        min_dist = torch.tensor(0.0, device=device, dtype=distances.dtype)
        max_dist = torch.max(distances)

        if max_dist < 1e-9:
            logging.warning(
                f"[Frame:{fn}] Max distance in ring division is near zero ({max_dist.item()}). All pixels might be in the first ring.")
            ring_bins = torch.zeros(nr + 1, device=device, dtype=distances.dtype)
            ring_bins[1:] = max_dist + 1e-6
        else:
            ring_bins = torch.linspace(min_dist.item(), (max_dist + 1e-6).item(), nr + 1, device=device,
                                       dtype=distances.dtype)

        ring_indices = torch.full_like(distances, -1, dtype=torch.long)

        for i in range(nr):
            lower_bound = ring_bins[i]
            upper_bound = ring_bins[i + 1]

            mask = (distances >= lower_bound) & (distances < upper_bound)
            if i == nr - 1:
                mask = (distances >= lower_bound) & (distances <= upper_bound)

            ring_indices[mask] = i

        if nr > 0 and ring_bins.numel() > 1:
            ring_indices[distances < ring_bins[1]] = 0

        rings: List[Optional[torch.Tensor]] = [None] * nr
        for rdx in range(nr):
            coords_tensor = torch.nonzero(ring_indices == rdx, as_tuple=False)
            if coords_tensor.numel() > 0:
                rings[rdx] = coords_tensor.long()
            else:
                logging.debug(f"[Frame:{fn}] Ring {rdx} is empty.")

        return rings
    except Exception as e:
        logging.error(f"Ring division PyTorch error Frame {fn}: {e}", exc_info=True)
        return [None] * nr


def calculate_entropies_torch(rv_tensor: torch.Tensor, fn: int = -1, ri: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет шенноновскую энтропию и "энтропию столкновений" (ваша формула)
    для одномерного PyTorch тензора значений пикселей (нормализованных).
    Все вычисления выполняются на устройстве rv_tensor.

    Args:
        rv_tensor: Одномерный PyTorch тензор значений пикселей, нормализованных в [0, 1].
        fn: Номер кадра (для логирования, опционально).
        ri: Индекс кольца (для логирования, опционально).

    Returns:
        Кортеж (torch.Tensor, torch.Tensor):
            - Шенноновская энтропия (скалярный тензор).
            - Энтропия столкновений (скалярный тензор).
            Возвращает (0.0, 0.0) тензоры, если тензор пуст или все его значения одинаковы.
    """
    eps = 1e-12
    shannon_entropy = torch.tensor(0.0, device=rv_tensor.device, dtype=rv_tensor.dtype)
    collision_entropy = torch.tensor(0.0, device=rv_tensor.device, dtype=rv_tensor.dtype)

    if rv_tensor.numel() > 0:
        if torch.all(rv_tensor == rv_tensor[0]):
            return shannon_entropy, collision_entropy

        # Для torch.histc нужен float тензор. Если rv_tensor уже float, это не повредит.
        # Если он другой (например, double), лучше привести явно.
        hist = torch.histc(rv_tensor.float(), bins=256, min=0.0, max=1.0)

        total_count = rv_tensor.numel()
        if total_count > 0:
            probabilities = hist / total_count

            p_mask = probabilities > eps
            p = probabilities[p_mask]

            if p.numel() > 0:
                shannon_entropy = -torch.sum(p * torch.log2(p))

                one_t = torch.tensor(1.0, device=p.device, dtype=p.dtype)
                ee = -torch.sum(p * torch.exp(one_t - p))
                collision_entropy = ee

    return shannon_entropy, collision_entropy


# def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
#     """
#         Вычисляет шенноновскую энтропию и энтропию столкновений (по вашей старой формуле)
#         для одномерного NumPy массива значений пикселей (предположительно, нормализованных).
#
#         Эта функция используется для оценки текстурности или сложности области (кольца)
#         с целью выбора наиболее подходящих областей для встраивания ЦВЗ.
#
#         Args:
#             rv: Одномерный NumPy массив значений пикселей, нормализованных в диапазоне [0, 1].
#             fn: Номер кадра (для логирования, опционально).
#             ri: Индекс кольца (для логирования, опционально).
#
#         Returns:
#             Кортеж (float, float):
#                 - Шенноновская энтропия.
#                 - Энтропия столкновений (согласно вашей предыдущей реализации).
#                 Возвращает (0.0, 0.0), если массив пуст или все его значения одинаковы.
#         """
#     eps = 1e-12;
#     shannon_entropy = 0.;
#     collision_entropy = 0.
#
#     if rv.size > 0:
#         rv_processed = np.clip(rv.copy(), 0.0, 1.0)
#         if np.all(rv_processed == rv_processed[0]): return 0.0, 0.0
#
#         hist, _ = np.histogram(rv_processed, bins=256, range=(0., 1.), density=False)
#         total_count = rv_processed.size
#         if total_count > 0:
#             probabilities = hist / total_count
#             p = probabilities[probabilities > eps]
#             if p.size > 0:
#                 shannon_entropy = -np.sum(p * np.log2(p))
#                 ee = -np.sum(p * np.exp(1. - p))
#                 collision_entropy = ee
#     return shannon_entropy, collision_entropy

def compute_adaptive_alpha_entropy_torch(
        rv_tensor: torch.Tensor,
        ri: int,
        fn: int,
        alpha_min_val: float = ALPHA_MIN,
        alpha_max_val: float = ALPHA_MAX,
        max_theoretical_entropy_val: float = MAX_THEORETICAL_ENTROPY
) -> torch.Tensor:
    """
    Вычисляет адаптивный коэффициент силы встраивания (альфа) на основе
    энтропии и дисперсии значений пикселей в указанном кольце, используя PyTorch.

    Args:
        rv_tensor: Одномерный PyTorch тензор значений пикселей кольца (нормализованных).
        ri: Индекс кольца (для логирования).
        fn: Номер кадра (для логирования).
        alpha_min_val, alpha_max_val, max_theoretical_entropy_val: Параметры.

    Returns:
        torch.Tensor: Адаптивный коэффициент альфа (скалярный тензор).
    """
    alpha_min_t = torch.tensor(alpha_min_val, device=rv_tensor.device, dtype=rv_tensor.dtype)
    alpha_max_t = torch.tensor(alpha_max_val, device=rv_tensor.device, dtype=rv_tensor.dtype)
    max_theoretical_entropy_t = torch.tensor(max_theoretical_entropy_val, device=rv_tensor.device,
                                             dtype=rv_tensor.dtype)

    if rv_tensor.numel() < 10:
        return alpha_min_t

    ve_tensor, _ = calculate_entropies_torch(rv_tensor.float(), fn, ri)  # Убедимся, что float
    lv_tensor = torch.var(rv_tensor.float())

    if not torch.isfinite(ve_tensor) or not torch.isfinite(lv_tensor):
        logging.warning(
            f"[F:{fn}, R:{ri}] PyTorch: Non-finite entropy ({ve_tensor.item()}) or variance ({lv_tensor.item()}). Using ALPHA_MIN.")
        return alpha_min_t

    en_tensor = torch.clip(ve_tensor / (max_theoretical_entropy_t + 1e-9), 0., 1.)

    vmp_t = torch.tensor(0.005, device=rv_tensor.device, dtype=rv_tensor.dtype)
    vsc_t = torch.tensor(500.0, device=rv_tensor.device, dtype=rv_tensor.dtype)

    exp_term_tensor = torch.exp(-vsc_t * (lv_tensor - vmp_t))

    one_t = torch.tensor(1.0, device=rv_tensor.device, dtype=rv_tensor.dtype)
    tn_denominator = one_t + exp_term_tensor
    # Защита от деления на ноль, если (1+exp_term) может быть очень мал
    tn_tensor = torch.where(tn_denominator < 1e-9, one_t, one_t / tn_denominator)

    we_t = torch.tensor(0.6, device=rv_tensor.device, dtype=rv_tensor.dtype)
    wt_t = torch.tensor(0.4, device=rv_tensor.device, dtype=rv_tensor.dtype)

    mf_tensor = torch.clip((we_t * en_tensor + wt_t * tn_tensor), 0., 1.)

    fa_tensor = alpha_min_t + (alpha_max_t - alpha_min_t) * mf_tensor

    logging.debug(
        f"[F:{fn}, R:{ri}] PyTorch Alpha={fa_tensor.item():.4f} (E={ve_tensor.item():.3f},V={lv_tensor.item():.6f})")
    return torch.clip(fa_tensor, alpha_min_t, alpha_max_t)


# def compute_adaptive_alpha_entropy(rv: np.ndarray, ri: int, fn: int) -> float:
#     """
#         Вычисляет адаптивный коэффициент силы встраивания (альфа) на основе
#         энтропии и дисперсии значений пикселей в указанном кольце.
#
#         Более высокие значения альфы (ближе к ALPHA_MAX) используются для более
#         текстурированных/сложных областей, что позволяет встроить более сильный
#         (робастный) сигнал с меньшим риском визуальных искажений. Для гладких
#         областей используется меньшая альфа (ближе к ALPHA_MIN).
#
#         Args:
#             rv: Одномерный NumPy массив значений пикселей кольца (нормализованных).
#             ri: Индекс кольца (для логирования).
#             fn: Номер кадра (для логирования).
#
#         Returns:
#             float: Адаптивный коэффициент альфа в диапазоне [ALPHA_MIN, ALPHA_MAX].
#                    Возвращает ALPHA_MIN, если данных для статистики недостаточно
#                    или при ошибках вычисления.
#         """
#     if rv.size < 10: return ALPHA_MIN
#     ve, _ = calculate_entropies(rv, fn, ri)
#     lv = np.var(rv)
#     if not np.isfinite(ve) or not np.isfinite(lv):
#         logging.warning(f"[F:{fn}, R:{ri}] Non-finite entropy ({ve}) or variance ({lv}). Using ALPHA_MIN.")
#         return ALPHA_MIN
#
#     en = np.clip(ve / MAX_THEORETICAL_ENTROPY, 0., 1.)
#     vmp = 0.005;
#     vsc = 500
#     try:
#         exp_term = np.exp(-vsc * (lv - vmp))
#     except OverflowError:
#         exp_term = 0.0
#     tn = 1. / (1. + exp_term) if (1. + exp_term) != 0 else 1.0
#
#     we = .6;
#     wt = .4
#     mf = np.clip((we * en + wt * tn), 0., 1.)
#     fa = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * mf
#     logging.debug(f"[F:{fn}, R:{ri}] Alpha={fa:.4f} (E={ve:.3f},V={lv:.6f})")
#     return np.clip(fa, ALPHA_MIN, ALPHA_MAX)


def get_fixed_pseudo_random_rings(pi: int, nr: int, ps: int) -> List[int]:
    """
    Генерирует детерминированный псевдослучайный набор индексов колец.
    pi: Индекс пары (или другой уникальный идентификатор).
    nr: Общее количество колец (N_RINGS).
    ps: Размер пула кандидатов (CANDIDATE_POOL_SIZE).
    """
    if ps <= 0:
        logging.warning(f"Pool size {ps} <= 0, returning empty list.")
        return []
    if ps > nr:
        logging.warning(f"Pool size {ps} > N_RINGS {nr}. Clamping pool size to {nr}.")
        ps = nr

    #индекс пары как сид для PRNG
    seed_str = str(pi).encode('utf-8')
    hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big')
    prng = random.Random(seed_int)

    try:
        candidate_indices = prng.sample(range(nr), ps)
    except ValueError:
        if nr == 0:
            candidate_indices = []
        else:
            logging.warning(f"prng.sample failed for P:{pi}, nr:{nr}, ps:{ps}. Using shuffle fallback.")
            candidate_indices = list(range(nr))
            prng.shuffle(candidate_indices)
            candidate_indices = candidate_indices[:ps]

    # logging.debug(f"[P:{pi}] Candidate rings: {candidate_indices}")
    return candidate_indices

# --- calculate_perceptual_mask ---
def get_sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Создает ядра Собеля Gx и Gy для свертки."""
    # Gx
    kernel_x = torch.tensor([[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]], device=device, dtype=dtype)
    # Gy
    kernel_y = torch.tensor([[-1., -2., -1.],
                             [0., 0., 0.],
                             [1., 2., 1.]], device=device, dtype=dtype)
    # Для conv2d ядро должно быть (out_channels, in_channels, H, W)
    # У нас 1 входной канал (изображение) и 1 выходной (карта градиента)
    return kernel_x.unsqueeze(0).unsqueeze(0), kernel_y.unsqueeze(0).unsqueeze(0)


def get_gaussian_kernel(kernel_size: int = 11, sigma: float = 5.0, channels: int = 1,
                        device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Создает Гауссово ядро для свертки."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # Должно быть нечетным
        logging.warning(f"Размер Гауссова ядра был изменен на {kernel_size}, т.к. он должен быть нечетным.")

    # Создаем 1D Гауссово ядро
    x_cord = torch.arange(kernel_size, device=device, dtype=dtype)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Гауссова формула
    gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Нормализуем ядро, чтобы сумма была равна 1
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Делаем его совместимым с conv2d: (out_channels, in_channels/groups, H, W)
    # Мы хотим, чтобы каждый канал обрабатывался независимо, поэтому out_channels = in_channels = channels, groups = channels
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)  # (channels, 1, H, W)

    return gaussian_kernel


# --- Новая функция calculate_perceptual_mask_torch ---
def calculate_perceptual_mask_torch(
        ip_tensor: torch.Tensor,
        device: torch.device,
        fn: int = -1,
        sobel_kernel_x: Optional[torch.Tensor] = None,  # Для кэширования ядер
        sobel_kernel_y: Optional[torch.Tensor] = None,
        gaussian_kernel_conv: Optional[torch.Tensor] = None,
        gaussian_ksize: int = 11,  # Размер ядра Гаусса
        gaussian_sigma: float = 5.0,  # Сигма Гаусса
        lambda_param: float = LAMBDA_PARAM  # Используем глобальную LAMBDA_PARAM
) -> torch.Tensor:
    """
    Вычисляет перцептуальную маску для 2D тензора (один канал изображения) на GPU.
    ip_tensor должен быть 2D тензором (H, W) или 4D (B, C, H, W) с B=1, C=1.
    Значения в ip_tensor предполагаются в диапазоне [0, 1].
    """
    if not isinstance(ip_tensor, torch.Tensor):
        logging.error(f"Mask error F{fn}: Input is not a tensor.")
        # Возвращаем единичный тензор подходящей формы, если ip_tensor не тензор, или его форма неизвестна
        # Для безопасности, лучше чтобы вызывающий код гарантировал корректный ip_tensor
        raise ValueError("Input to calculate_perceptual_mask_torch must be a PyTorch tensor.")

    original_ndim = ip_tensor.ndim
    if original_ndim == 2:
        # (H, W) -> (1, 1, H, W) для conv2d
        tensor_4d = ip_tensor.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 4 and ip_tensor.shape[0] == 1 and ip_tensor.shape[1] == 1:
        tensor_4d = ip_tensor
    else:
        logging.error(f"Mask error F{fn}: Input tensor must be 2D (H,W) or 4D (1,1,H,W), got {ip_tensor.shape}")
        # Возвращаем единичный тензор, пытаясь сохранить исходную форму если возможно
        return torch.ones_like(ip_tensor, device=device)

    if not torch.is_floating_point(tensor_4d):
        logging.warning(f"Mask warning F{fn}: Input tensor is not float, converting to float32.")
        tensor_4d = tensor_4d.to(dtype=torch.float32)

    if not torch.all(torch.isfinite(tensor_4d)):
        logging.warning(
            f"Mask error F{fn}: Input tensor contains NaN/inf. Clipping might be needed or result unreliable.")
        # Можно добавить .clamp_(0, 1) здесь, если значения могут выходить за пределы
        # tensor_4d = torch.nan_to_num(tensor_4d, nan=0.0, posinf=1.0, neginf=0.0) # Замена нечисловых

    try:
        # --- 1. Градиенты Собеля ---
        if sobel_kernel_x is None or sobel_kernel_y is None:  # Создаем ядра, если не переданы (для кэширования)
            kernel_gx, kernel_gy = get_sobel_kernels(device, tensor_4d.dtype)
        else:
            kernel_gx, kernel_gy = sobel_kernel_x.to(device, tensor_4d.dtype), sobel_kernel_y.to(device,
                                                                                                 tensor_4d.dtype)

        # padding='same' сохраняет размер изображения
        # F.conv2d ожидает input (B, C_in, H_in, W_in) и weight (C_out, C_in/groups, H_k, W_k)
        grad_x = F.conv2d(tensor_4d, kernel_gx, padding='same')
        grad_y = F.conv2d(tensor_4d, kernel_gy, padding='same')

        if not torch.all(torch.isfinite(grad_x)) or not torch.all(torch.isfinite(grad_y)):
            logging.warning(f"Mask warning F{fn}: Sobel gradients contain NaN/inf.")
            # Заменяем NaN/Inf на 0, чтобы не прерывать вычисления
            grad_x = torch.nan_to_num(grad_x, nan=0.0)
            grad_y = torch.nan_to_num(grad_y, nan=0.0)

        # Величина градиента
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-12)  # + eps для стабильности sqrt

        # --- 2. Локальная дисперсия с Гауссовым размытием ---
        # E[X^2] - (E[X])^2
        if gaussian_kernel_conv is None:  # Создаем ядро, если не передано
            # Для Гаусса channels=1, так как мы работаем с одноканальным изображением яркости
            gaussian_kernel_conv = get_gaussian_kernel(kernel_size=gaussian_ksize, sigma=gaussian_sigma, channels=1,
                                                       device=device, dtype=tensor_4d.dtype)
        else:
            gaussian_kernel_conv = gaussian_kernel_conv.to(device, tensor_4d.dtype)

        # padding='same' для Гаусса
        # F.conv2d с groups=channels если ядро имеет channels=1 и мы хотим применить его к каждому каналу независимо.
        # Но здесь у нас 1 входной канал для tensor_4d (яркость), и ядро Гаусса тоже для 1 канала.
        local_mean = F.conv2d(tensor_4d, gaussian_kernel_conv, padding='same')
        local_mean_sq = F.conv2d(tensor_4d ** 2, gaussian_kernel_conv, padding='same')

        if not torch.all(torch.isfinite(local_mean)) or not torch.all(torch.isfinite(local_mean_sq)):
            logging.warning(f"Mask warning F{fn}: Gaussian blur results contain NaN/inf.")
            local_mean = torch.nan_to_num(local_mean, nan=0.0)
            local_mean_sq = torch.nan_to_num(local_mean_sq, nan=0.0)

        local_variance = local_mean_sq - local_mean ** 2
        # Защита от отрицательных значений из-за ошибок округления и от NaN в sqrt
        local_std_dev = torch.sqrt(torch.relu(local_variance) + 1e-12)

        # --- 3. Комбинированная сложность ---
        complexity_map = torch.maximum(gradient_magnitude, local_std_dev)

        # --- 4. Нормализация маски ---
        min_complexity = torch.min(complexity_map)  # Можно использовать torch.amin(complexity_map)
        max_complexity = torch.max(complexity_map)  # Можно использовать torch.amax(complexity_map)

        # Нормализация в [0, 1]
        # Защита от деления на ноль, если max_complexity == min_complexity (плоское изображение)
        if (max_complexity - min_complexity) < 1e-9:
            normalized_mask = torch.zeros_like(complexity_map, device=device)
        else:
            normalized_mask = (complexity_map - min_complexity) / (max_complexity - min_complexity)

        # Финальное клиппирование (хотя после нормализации должно быть в [0,1])
        final_mask = torch.clip(normalized_mask, 0., 1.)

        # Возвращаем маску в той же размерности, что и исходный ip_tensor (если он был 2D)
        if original_ndim == 2:
            return final_mask.squeeze(0).squeeze(0)  # (1,1,H,W) -> (H,W)
        else:  # Если исходный был 4D (1,1,H,W), возвращаем так же
            return final_mask

    except Exception as e:
        logging.error(f"Perceptual mask (PyTorch) general error F{fn}: {e}", exc_info=True)
        # Возвращаем единичную маску той же формы, что и входной тензор
        return torch.ones_like(ip_tensor.view_as(tensor_4d) if original_ndim == 2 else ip_tensor, device=device)


def add_ecc(data_bits: np.ndarray, bch_code: Optional[galois.BCH]) -> Optional[np.ndarray]:
    """
        Добавляет биты коррекции ошибок (ECC) к предоставленному массиву информационных бит
        с использованием указанного объекта кода BCH (Bose-Chaudhuri-Hocquenghem).

        Если ECC недоступен или объект bch_code не предоставлен, функция возвращает
        исходные биты данных без изменений (если это допустимо по логике).

        Args:
            data_bits: Одномерный NumPy массив информационных бит (0 или 1, dtype=uint8).
            bch_code: Объект galois.BCH, используемый для кодирования.
                      Если None, ECC не применяется.

        Returns:
            Optional[np.ndarray]: NumPy массив, содержащий кодовое слово (информационные биты + биты ECC).
                                  Возвращает исходные data_bits, если ECC не применен.
                                  Возвращает None в случае ошибки (например, если длина данных
                                  превышает возможности кода k).
        """
    if not GALOIS_AVAILABLE or bch_code is None:
        logging.warning("ECC не доступен или не предоставлен, возвращаем исходные биты.")
        return data_bits
    try:
        k = bch_code.k;
        n = bch_code.n
        if data_bits.size > k:
            logging.error(f"ECC Error: Data size ({data_bits.size}) > k ({k})")
            return None
        pad_len = k - data_bits.size

        msg_bits = data_bits.astype(np.uint8).flatten()
        if pad_len > 0:
            msg_bits = np.pad(msg_bits, (0, pad_len), 'constant')

        GF = bch_code.field;
        msg_vec = GF(msg_bits);
        cw_vec = bch_code.encode(msg_vec)
        pkt_bits = cw_vec.view(np.ndarray).astype(np.uint8)

        if pkt_bits.size != n:
            logging.error(f"ECC Error: Output packet size ({pkt_bits.size}) != n ({n})")
            return None
        logging.info(f"Galois ECC: Data({data_bits.size}b->{k}b) -> Packet({pkt_bits.size}b).")
        return pkt_bits
    except Exception as e:
        logging.error(f"Galois encode error: {e}", exc_info=True)
        return None


CODEC_CONTAINER_COMPATIBILITY: Dict[str, Set[Tuple[str, str]]] = {
    ".mp4": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('audio', 'aac'), ('audio', 'mp3'),
             ('audio', 'alac')},
    ".mov": {('video', 'h264'), ('video', 'hevc'), ('video', 'mpeg4'), ('video', 'prores'), ('audio', 'aac'),
             ('audio', 'mp3'), ('audio', 'alac'), ('audio', 'pcm_s16le')},
    ".mkv": {('video', 'h264'), ('video', 'hevc'), ('video', 'vp9'), ('video', 'av1'), ('video', 'mpeg4'),
             ('video', 'mpeg2video'), ('video', 'theora'), ('video', 'prores'), ('audio', 'aac'), ('audio', 'opus'),
             ('audio', 'vorbis'), ('audio', 'flac'), ('audio', 'ac3'), ('audio', 'dts'), ('audio', 'mp3'),
             ('audio', 'pcm_s16le'), ('audio', 'alac')},
    ".webm": {('video', 'vp8'), ('video', 'vp9'), ('video', 'av1'), ('audio', 'opus'), ('audio', 'vorbis')},
}
DEFAULT_OUTPUT_CONTAINER_EXT_FINAL: str = ".mp4"
DEFAULT_VIDEO_ENCODER_LIB_FOR_HEAD: str = "libx264"
DEFAULT_VIDEO_CODEC_NAME_FOR_HEAD: str = "h264"
DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL: str = "aac"
FALLBACK_CONTAINER_EXT_FINAL: str = ".mkv"


def get_iframe_start_times(filepath: str) -> List[float]:
    """
    Анализирует видеофайл и возвращает список временных меток (PTS в секундах)
    всех ключевых кадров (I-frames) в видеопотоке.

    Args:
        filepath: Путь к видеофайлу.

    Returns:
        Список float: временные метки начала I-кадров в секундах,
                      отсортированный по возрастанию. Пустой список, если
                      видеопоток не найден, I-кадры не найдены, или произошла ошибка.
    """
    if not PYAV_AVAILABLE:
        logging.error("PyAV недоступен. Невозможно получить времена I-кадров.")
        return []

    iframe_times_sec: List[float] = []
    container: Optional[av.container.Container] = None

    try:
        logging.debug(f"Анализ I-кадров в файле: '{filepath}'")
        container = av.open(filepath, mode='r', metadata_errors='ignore')
        if container is None:
            logging.error(f"Не удалось открыть файл '{filepath}' с помощью PyAV для поиска I-кадров.")
            return []

        video_stream = next((s for s in container.streams if s.type == 'video'), None)

        if not video_stream:
            logging.warning(f"Видеопоток не найден в файле '{filepath}'.")
            return []

        if video_stream.time_base is None:
            logging.error(f"Отсутствует time_base для видеопотока в '{filepath}'. Невозможно рассчитать времена.")
            return []

        if video_stream.time_base.denominator == 0:
            logging.error(f"Некорректный time_base (знаменатель 0) для видеопотока в '{filepath}'.")
            return []

        logging.debug(
            f"Видеопоток найден: index={video_stream.index}, codec={video_stream.name}, time_base={video_stream.time_base}")


        key_packet_pts_set = set()

        for packet in container.demux(video_stream):
            if packet.stream.type != 'video' or packet.stream.index != video_stream.index:
                continue

            if packet.is_keyframe and packet.pts is not None:
                key_packet_pts_set.add(packet.pts)

        if key_packet_pts_set:
            iframe_times_sec = sorted([float(pts * video_stream.time_base) for pts in key_packet_pts_set])
            logging.info(
                f"Найдено {len(iframe_times_sec)} уникальных I-кадров (по пакетам). Первые 5: {[f'{t:.3f}s' for t in iframe_times_sec[:5]]}")
        else:
            logging.warning(f"I-кадры (по пакетам) не найдены в '{filepath}'.")

    except av.FFmpegError as e_av:
        logging.error(f"Ошибка PyAV/FFmpeg при поиске I-кадров в '{filepath}': {e_av}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"Неожиданная ошибка при поиске I-кадров в '{filepath}': {e}", exc_info=True)
        return []
    finally:
        if container:
            try:
                container.close()
            except av.FFmpegError as e_close:
                logging.error(f"Ошибка при закрытии контейнера '{filepath}' после поиска I-кадров: {e_close}")

    return iframe_times_sec

def check_compatibility_and_choose_output(
        input_metadata: Dict[str, Any]
) -> Tuple[str, str, str]:
    """
    Анализирует метаданные входа, выбирает:
    1. Рекомендуемую библиотеку CPU-кодера для "головы" (для записи в temp_head).
    2. Расширение для ФИНАЛЬНОГО выходного файла.
    3. Рекомендуемое действие для аудиопотока "хвоста" при склейке FFmpeg
       ('copy', 'none', или имя кодека для перекодирования, e.g., 'aac').

    Returns:
        Tuple[str, str, str]:
            - final_output_extension (e.g., ".mp4")
            - recommended_head_video_encoder_lib (e.g., "libx264")
            - ffmpeg_tail_audio_action (e.g., "copy", "none", "aac")
    """
    in_video_codec = input_metadata.get('video_codec')  # e.g., 'h264', 'hevc'
    in_audio_codec = input_metadata.get('audio_codec') if input_metadata.get('has_audio') else None
    has_audio_original = input_metadata.get('has_audio', False)

    logging.info(f"Проверка совместимости и выбор параметров выхода:")
    logging.info(f"  Вход: Видео='{in_video_codec}', Аудио='{in_audio_codec}' (есть: {has_audio_original})")

    # --- 1. Выбор рекомендуемого CPU-кодера для "головы" (temp_head.mp4) ---
    recommended_head_video_encoder_lib = DEFAULT_VIDEO_ENCODER_LIB_FOR_HEAD

    logging.info(f"  Рекомендуемый видеокодер для 'головы' (temp_file): {recommended_head_video_encoder_lib}")

    head_audio_codec_name_assumed = 'aac'

    # --- 2. Определение расширения для ФИНАЛЬНОГО файла и действия для аудио "хвоста" (в FFmpeg) ---

    final_output_extension = DEFAULT_OUTPUT_CONTAINER_EXT_FINAL  # По умолчанию .mp4

    ffmpeg_tail_audio_action = DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL
    if not has_audio_original:
        ffmpeg_tail_audio_action = 'none'
    elif in_audio_codec:
        if in_audio_codec == 'aac' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.mp4', '.mov', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
        elif in_audio_codec == 'mp3' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.mp4', '.mov', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
        elif in_audio_codec == 'opus' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.webm', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
            final_output_extension = ".webm" if in_video_codec in ['vp8', 'vp9', 'av1'] else ".mkv"
        elif in_audio_codec == 'vorbis' and (DEFAULT_OUTPUT_CONTAINER_EXT_FINAL in ['.webm', '.mkv']):
            ffmpeg_tail_audio_action = 'copy'
            final_output_extension = ".webm" if in_video_codec in ['vp8', 'vp9', 'av1'] else ".mkv"

    if in_video_codec in ['vp8', 'vp9', 'av1']:
        if final_output_extension == ".mp4":
            final_output_extension = ".mkv"
            if ffmpeg_tail_audio_action not in ['opus', 'vorbis', 'aac', 'none', 'copy']:
                ffmpeg_tail_audio_action = DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL


    final_check_audio_codec = in_audio_codec if ffmpeg_tail_audio_action == 'copy' else ffmpeg_tail_audio_action
    if final_check_audio_codec == 'none': final_check_audio_codec = None  # Для проверки

    allowed_codecs_in_final = CODEC_CONTAINER_COMPATIBILITY.get(final_output_extension, set())

    video_ok_final = (in_video_codec is None) or (('video', in_video_codec) in allowed_codecs_in_final)
    audio_head_ok_final = ('audio', head_audio_codec_name_assumed) in allowed_codecs_in_final
    audio_tail_ok_final = (final_check_audio_codec is None) or \
                          (('audio', final_check_audio_codec) in allowed_codecs_in_final)

    if not (video_ok_final and audio_head_ok_final and audio_tail_ok_final):
        logging.warning(
            f"  Выбранный финальный контейнер '{final_output_extension}' может быть несовместим с кодеками:")
        logging.warning(f"    Видео хвоста ({in_video_codec or 'N/A'}): {video_ok_final}")
        logging.warning(f"    Аудио головы ({head_audio_codec_name_assumed}): {audio_head_ok_final}")
        logging.warning(f"    Аудио хвоста ({final_check_audio_codec or 'N/A'}): {audio_tail_ok_final}")

        current_problematic_ext = final_output_extension
        final_output_extension = FALLBACK_CONTAINER_EXT_FINAL
        logging.warning(f"  Переключение на fallback финальный контейнер: '{final_output_extension}'")

        allowed_codecs_in_mkv = CODEC_CONTAINER_COMPATIBILITY.get(".mkv", set())
        video_ok_mkv = (in_video_codec is None) or (('video', in_video_codec) in allowed_codecs_in_mkv)
        audio_head_ok_mkv = ('audio', head_audio_codec_name_assumed) in allowed_codecs_in_mkv
        audio_tail_ok_mkv = (final_check_audio_codec is None) or \
                            (('audio', final_check_audio_codec) in allowed_codecs_in_mkv)

        if not (video_ok_mkv and audio_head_ok_mkv and audio_tail_ok_mkv):
            logging.error(f"  Даже fallback контейнер '{final_output_extension}' несовместим! "
                          f"Возврат к '{current_problematic_ext}'. FFmpeg может потребовать перекодирование или выдать ошибку.")
            final_output_extension = current_problematic_ext
            if ffmpeg_tail_audio_action == 'copy' and has_audio_original:
                logging.warning(
                    f"   Аудио хвоста будет принудительно перекодировано в {DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL} из-за несовместимости контейнера.")
                ffmpeg_tail_audio_action = DEFAULT_AUDIO_CODEC_FOR_FFMPEG_TAIL

    logging.info(f"  Итоговое решение для финального файла: Расширение='{final_output_extension}', "
                 f"Видеокодер 'головы' (для temp файла)='{recommended_head_video_encoder_lib}', "
                 f"Действие для аудио 'хвоста' (FFmpeg)='{ffmpeg_tail_audio_action}'")

    return final_output_extension, recommended_head_video_encoder_lib, ffmpeg_tail_audio_action


# --- Функция чтения видео ---
def get_input_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Читает метаданные из видеофайла с использованием PyAV, включая битрейт аудио
    и индексы потоков. Предоставляет fallback на OpenCV для базовых параметров.

    Args:
        video_path: Путь к видеофайлу.

    Returns:
        Словарь с метаданными или None при критической ошибке.
        Ключи словаря включают:
        'input_path', 'format_name', 'duration' (в микросекундах, если есть),
        'video_codec', 'width', 'height', 'fps' (Fraction или float),
        'video_bitrate', 'video_time_base' (Fraction), 'pix_fmt',
        'color_space_tag', 'color_primaries_tag', 'color_transfer_tag', 'color_range_tag',
        'video_stream_index' (int),
        'has_audio' (bool), 'audio_codec', 'audio_rate' (int), 'audio_layout' (str),
        'audio_time_base' (Fraction), 'audio_bitrate' (int, bps, или None),
        'audio_stream_index' (int, или -1),
        'audio_codec_context_params' (dict),
        'total_frames' (int, оценка, может быть неточной).
    """
    if not PYAV_AVAILABLE:
        logging.error("PyAV недоступен для get_input_metadata.")
        return None

    metadata: Dict[str, Any] = {
        'input_path': video_path,
        'format_name': None,
        'duration': None,
        'video_codec': None,
        'width': 0,
        'height': 0,
        'fps': None,
        'video_bitrate': None,
        'video_time_base': None,
        'pix_fmt': None,
        'color_space_tag': None,
        'color_primaries_tag': None,
        'color_transfer_tag': None,
        'color_range_tag': None,
        'video_stream_index': -1,
        'has_audio': False,
        'audio_codec': None,
        'audio_rate': None,
        'audio_layout': None,
        'audio_time_base': None,
        'audio_bitrate': None,
        'audio_stream_index': -1,
        'audio_codec_context_params': None,
        'total_frames': 0
    }
    input_container: Optional[av.container.Container] = None
    opencv_fallback_used = False

    try:
        logging.info(f"Attempting to open '{video_path}' with PyAV to read metadata...")
        try:
            input_container = av.open(video_path, mode='r')
            if input_container is None:
                raise av.FFmpegError(f"av.open вернул None для '{video_path}'")
            logging.info("Opened with PyAV successfully.")
        except (av.FFmpegError, FileNotFoundError, Exception) as e_open:
            logging.error(f"PyAV не смог открыть '{video_path}' для метаданных: {e_open}", exc_info=True)
            logging.warning("Попытка fallback на OpenCV для базовых метаданных (W, H, FPS)...")
            opencv_fallback_used = True
            cap_cv2 = None
            try:
                cap_cv2 = cv2.VideoCapture(video_path)
                if cap_cv2.isOpened():
                    w_cv2 = int(cap_cv2.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h_cv2 = int(cap_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps_cv2 = cap_cv2.get(cv2.CAP_PROP_FPS)
                    if w_cv2 > 0 and h_cv2 > 0:
                        metadata['width'] = w_cv2
                        metadata['height'] = h_cv2
                        logging.info(f"  OpenCV Fallback: Получены размеры {w_cv2}x{h_cv2}")
                    if fps_cv2 and fps_cv2 > 0:
                        metadata['fps'] = float(fps_cv2)  # Сохраняем как float
                        logging.info(f"  OpenCV Fallback: Получен FPS {fps_cv2:.2f}")
                else:
                    logging.error("  OpenCV fallback не удался: Не удалось открыть видео.")
            except Exception as e_cv2_meta:
                logging.error(f"  Ошибка во время OpenCV fallback: {e_cv2_meta}")
            finally:
                if cap_cv2: cap_cv2.release()
            if not (metadata['width'] > 0 and metadata['height'] > 0):
                logging.critical("Не удалось получить даже базовые размеры кадра.")
                return None
            return metadata

        # --- Чтение метаданных контейнера ---
        if input_container.format:
            metadata['format_name'] = input_container.format.name
        if input_container.duration:
            metadata['duration'] = input_container.duration
        if input_container.bit_rate:
            pass

        # --- Чтение метаданных видеопотока ---
        if not input_container.streams.video:
            logging.warning("Видеопотоки не найдены PyAV.")
            if not (metadata['width'] > 0 and metadata['height'] > 0):
                logging.error("Нет видеопотоков и не удалось получить размеры через OpenCV.")
                return None
        else:
            try:
                video_stream = input_container.streams.video[0]
                metadata['video_stream_index'] = video_stream.index
                ctx = video_stream.codec_context

                metadata['video_codec'] = video_stream.codec.name
                if ctx.width and ctx.width > 0: metadata['width'] = ctx.width
                if ctx.height and ctx.height > 0: metadata['height'] = ctx.height

                fps_val = None
                if video_stream.average_rate and float(video_stream.average_rate) > 0:
                    fps_val = video_stream.average_rate
                elif video_stream.r_frame_rate and float(video_stream.r_frame_rate) > 0:
                    fps_val = video_stream.r_frame_rate
                if fps_val: metadata['fps'] = fps_val

                metadata['pix_fmt'] = ctx.pix_fmt
                if ctx.bit_rate and ctx.bit_rate > 0:
                    metadata['video_bitrate'] = ctx.bit_rate
                elif input_container.bit_rate and input_container.bit_rate > 0 and not input_container.streams.audio:
                    metadata['video_bitrate'] = input_container.bit_rate

                if video_stream.time_base: metadata['video_time_base'] = video_stream.time_base

                # Цветовые теги
                metadata['color_space_tag'] = video_stream.metadata.get('color_space')
                metadata['color_primaries_tag'] = video_stream.metadata.get('color_primaries')
                metadata['color_transfer_tag'] = video_stream.metadata.get('color_transfer')
                metadata['color_range_tag'] = video_stream.metadata.get('color_range')

                # Оценка общего числа кадров
                if video_stream.frames and video_stream.frames > 0:
                    metadata['total_frames'] = video_stream.frames
                elif metadata['duration'] and metadata['fps'] and float(metadata['fps']) > 0:
                    try:
                        metadata['total_frames'] = int(
                            round((float(metadata['duration']) / 1_000_000.0) * float(metadata['fps'])))
                    except Exception:
                        pass

                logging.info(
                    f"  PyAV Video Stream Meta: Codec={metadata['video_codec']}, Res={metadata['width']}x{metadata['height']}, "
                    f"FPS={float(metadata['fps']):.2f if metadata['fps'] else 'N/A'}, Frames={metadata['total_frames'] or 'N/A'}, "
                    f"Index={metadata['video_stream_index']}")

            except (AttributeError, ValueError, TypeError, av.FFmpegError) as e_video:
                logging.error(f"Ошибка при доступе к свойствам видеопотока: {e_video}")
                if not (metadata['width'] > 0 and metadata['height'] > 0 and metadata['fps']):
                    logging.critical("Критичные видео метаданные отсутствуют после всех проверок.")
                    return None

        # --- Чтение метаданных аудиопотока ---
        if not input_container.streams.audio:
            logging.info("Аудиопотоки не найдены PyAV.")
            metadata['has_audio'] = False
            metadata['audio_stream_index'] = -1
            metadata['audio_bitrate'] = None
        else:
            try:
                audio_stream = input_container.streams.audio[0]
                metadata['audio_stream_index'] = audio_stream.index
                ctx = audio_stream.codec_context

                metadata['has_audio'] = True
                metadata['audio_codec'] = audio_stream.codec.name
                metadata['audio_rate'] = ctx.rate
                metadata['audio_layout'] = ctx.layout.name if ctx.layout else None
                if audio_stream.time_base: metadata['audio_time_base'] = audio_stream.time_base

                # --- Получение аудио битрейта ---
                audio_bitrate = ctx.bit_rate
                if audio_bitrate and audio_bitrate > 0:
                    metadata['audio_bitrate'] = audio_bitrate
                else:
                    metadata['audio_bitrate'] = None

                metadata['audio_codec_context_params'] = {
                    'format': ctx.format.name if ctx.format else None,
                    'layout': metadata['audio_layout'],
                    'rate': metadata['audio_rate'],
                    'bit_rate': metadata['audio_bitrate'],
                    'codec_tag': ctx.codec_tag,
                    'extradata': bytes(ctx.extradata) if ctx.extradata else None,
                }
                logging.info(
                    f"  PyAV Audio Stream Meta: Codec={metadata['audio_codec']}, Rate={metadata['audio_rate']}, "
                    f"Layout={metadata['audio_layout']}, Bitrate={metadata['audio_bitrate'] or 'N/A'}, "
                    f"Index={metadata['audio_stream_index']}")

            except (AttributeError, ValueError, TypeError, av.FFmpegError) as e_audio:
                logging.warning(f"Не удалось получить/обработать метаданные аудиопотока: {e_audio}")
                metadata['has_audio'] = False
                metadata['audio_stream_index'] = -1
                metadata['audio_bitrate'] = None

        if not (metadata['width'] > 0 and metadata['height'] > 0 and metadata['fps']):
            logging.critical("Критичные видео метаданные (W, H, FPS) отсутствуют.")
            return None

        return metadata

    except Exception as e_general:
        logging.error(f"Неожиданная ошибка при получении метаданных для '{video_path}': {e_general}", exc_info=True)
        return None
    finally:
        if input_container:
            try:
                input_container.close()
                logging.debug("Metadata reading: Input container closed.")
            except av.FFmpegError as e_close:
                logging.error(f"Error closing input container after metadata read: {e_close}")


# @profile
def read_processing_head(
        video_path: str,
        frames_to_read: int,
        video_stream_index: int,
        audio_stream_index: int
) -> Tuple[Optional[List[np.ndarray]], Optional[List[av.Packet]]]:
    """
    Читает и декодирует ТОЛЬКО первые `frames_to_read` видеокадров (в BGR NumPy)
    и собирает ВСЕ аудиопакеты из указанных потоков.

    Args:
        video_path: Путь к входному видеофайлу.
        frames_to_read: Количество видеокадров, которые нужно прочитать и декодировать.
        video_stream_index: Индекс видеопотока в контейнере.
        audio_stream_index: Индекс аудиопотока в контейнере (-1, если аудио не нужно/нет).

    Returns:
        Кортеж (list_of_bgr_frames, list_of_audio_packets).
        list_of_bgr_frames: Список NumPy массивов (кадры в BGR).
        list_of_audio_packets: Список всех av.Packet аудиопотока.
        Возвращает (None, None) при критической ошибке.
    """
    if not PYAV_AVAILABLE:
        logging.error("PyAV недоступен для read_processing_head.")
        return None, None

    if frames_to_read <= 0:
        logging.warning(
            f"read_processing_head: Количество кадров для чтения ({frames_to_read}) <= 0. Возвращаем пустые списки.")
        return [], []

    head_frames_bgr: List[np.ndarray] = []
    all_audio_packets: List[av.Packet] = []
    input_container: Optional[av.container.Container] = None
    frames_decoded_count = 0
    reformatter_yuv_to_bgr: Optional[VideoReformatter] = None

    logging.info(f"Чтение 'головы': {frames_to_read} видеокадров и все аудиопакеты из '{video_path}'...")
    logging.debug(f"  Целевой видеопоток: индекс {video_stream_index}, аудиопоток: индекс {audio_stream_index}")

    try:
        input_container = av.open(video_path, mode='r')
        if input_container is None:
            raise av.FFmpegError(f"av.open вернул None для файла: {video_path}")

        has_target_video_stream = any(
            s.index == video_stream_index and s.type == 'video' for s in input_container.streams)
        has_target_audio_stream = audio_stream_index != -1 and any(
            s.index == audio_stream_index and s.type == 'audio' for s in input_container.streams)

        if not has_target_video_stream and frames_to_read > 0:
            logging.error(f"  Видеопоток с индексом {video_stream_index} не найден в '{video_path}'.")
            return None, None
        if audio_stream_index != -1 and not has_target_audio_stream:
            logging.warning(f"  Аудиопоток с индексом {audio_stream_index} не найден. Аудиопакеты не будут собраны.")

        logging.debug("Начало демультиплексирования пакетов...")
        packet_count = 0
        for packet in input_container.demux():
            packet_count += 1
            if packet.dts is None:
                logging.debug(f"  Пакет {packet_count}: пропущен (нет DTS). Stream index: {packet.stream.index}")
                continue

            # сборка ВСЕх аудиопакеты нужного потока
            if has_target_audio_stream and packet.stream.index == audio_stream_index:
                try:
                    packet_data = bytes(packet)
                    new_packet = av.Packet(packet_data)

                    new_packet.pts = packet.pts
                    new_packet.dts = packet.dts
                    new_packet.duration = packet.duration

                    all_audio_packets.append(new_packet)
                except Exception as e_packet_create:
                    logging.error(f"  Ошибка при создании копии аудиопакета: {e_packet_create}", exc_info=True)

                if len(all_audio_packets) % 200 == 0:
                    logging.debug(f"  Собрано {len(all_audio_packets)} аудиопакетов...")

            # Декодирование видеопакетов, пока не набирается нужное количество кадров
            elif has_target_video_stream and packet.stream.index == video_stream_index:
                if frames_decoded_count >= frames_to_read:
                    continue

                try:
                    for frame in packet.decode():
                        if frame and isinstance(frame, av.VideoFrame):
                            try:
                                np_frame_bgr = frame.to_ndarray(format='bgr24')
                            except (av.FFmpegError, ValueError, TypeError) as e_to_ndarray:
                                logging.warning(
                                    f"    Не удалось напрямую конвертировать видеокадр в bgr24: {e_to_ndarray}. Попытка через YUV420p.")
                                try:
                                    if reformatter_yuv_to_bgr is None:
                                        if frame.width <= 0 or frame.height <= 0:
                                            logging.error(
                                                f"    Невалидные размеры кадра для реформаттера: {frame.width}x{frame.height}")
                                            continue
                                        reformatter_yuv_to_bgr = VideoReformatter(frame.width, frame.height, 'yuv420p')

                                    frame_yuv = reformatter_yuv_to_bgr.reformat(frame)
                                    np_frame_yuv = frame_yuv.to_ndarray()

                                    if np_frame_yuv.shape[
                                        0] * 2 // 3 == frame_yuv.height:
                                        np_frame_bgr = cv2.cvtColor(np_frame_yuv, cv2.COLOR_YUV2BGR_I420)
                                    else:
                                        logging.error(
                                            f"    Неизвестный формат NumPy массива после YUV реформатирования: {np_frame_yuv.shape}")
                                        continue
                                except Exception as e_reformat_cv:
                                    logging.error(
                                        f"    Ошибка при реформатировании в YUV или конвертации OpenCV: {e_reformat_cv}",
                                        exc_info=True)
                                    continue

                            head_frames_bgr.append(np_frame_bgr)
                            frames_decoded_count += 1

                            if frames_decoded_count % 50 == 0:
                                logging.debug(
                                    f"  Декодировано {frames_decoded_count}/{frames_to_read} видеокадров 'головы'...")

                            if frames_decoded_count >= frames_to_read:
                                break

                    if frames_decoded_count >= frames_to_read:
                        pass

                except (av.FFmpegError, ValueError) as e_decode_video:
                    logging.warning(
                        f"  Ошибка декодирования видеопакета (stream {packet.stream.index}): {e_decode_video} - пакет пропущен.")
                except Exception as e_unexpected_decode:
                    logging.error(f"  Неожиданная ошибка при декодировании видеопакета: {e_unexpected_decode}",
                                  exc_info=True)

            if frames_to_read > 0 and has_target_video_stream and frames_decoded_count >= frames_to_read and \
                    (audio_stream_index == -1 or not has_target_audio_stream):
                logging.info(
                    "  Все необходимые видеокадры 'головы' прочитаны, аудио не требуется/нет. Завершение демультиплексирования.")
                break

        if frames_to_read > 0 and frames_decoded_count < frames_to_read:
            logging.warning(
                f"  Прочитано только {frames_decoded_count} видеокадров 'головы', запрашивалось {frames_to_read} (возможно, конец файла).")

    except FFmpegEOFError:
        logging.info("  Достигнут конец файла (EOF) при чтении 'головы'.")
    except av.FFmpegError as e_av:
        logging.error(f"Ошибка PyAV/FFmpeg при чтении 'головы' из '{video_path}': {e_av}", exc_info=True)
        return None, None
    except FileNotFoundError:
        logging.error(f"Файл не найден при попытке открыть для чтения 'головы': '{video_path}'")
        return None, None
    except Exception as e_main:
        logging.error(f"Неожиданная ошибка при чтении 'головы' из '{video_path}': {e_main}", exc_info=True)
        return None, None
    finally:
        if input_container:
            try:
                input_container.close()
                logging.debug("  Контейнер входного файла для 'головы' закрыт.")
            except av.FFmpegError as e_close:
                logging.error(f"  Ошибка при закрытии контейнера входного файла 'головы': {e_close}")

    logging.info(
        f"Чтение 'головы' завершено. Декодировано {len(head_frames_bgr)} видеокадров. Собрано {len(all_audio_packets)} аудиопакетов.")
    return head_frames_bgr, all_audio_packets


# @profile
def rescale_time(value: Optional[int], old_tb: Optional[Fraction], new_tb: Optional[Fraction], label: str = "") -> Optional[int]:
    """
    Пересчитывает значение времени (PTS, DTS, Duration) из одной time_base в другую.
    Возвращает None при ошибке или если входные данные некорректны.
    """
    if value is None or old_tb is None or new_tb is None \
       or not isinstance(old_tb, Fraction) or not isinstance(new_tb, Fraction) \
       or old_tb.denominator == 0 or new_tb.denominator == 0:
         return None
    try:
         scaled_value = Fraction(value * old_tb.numerator * new_tb.denominator, old_tb.denominator * new_tb.numerator)
         if scaled_value >= 0: result = int(scaled_value + Fraction(1, 2))
         else: result = int(scaled_value - Fraction(1, 2))
         return result
    except (ZeroDivisionError, OverflowError, TypeError) as e:
         logging.warning(f"Rescale warning ({label}): value={value}, old={old_tb}, new={new_tb}. Error: {e}")
         return None

# --- Основная функция записи "Голова + Хвост" ---
def get_assumed_color_properties(width: int, height: int,
                                 original_tags: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Определяет предполагаемые/стандартные цветовые свойства на основе разрешения
    и имеющихся оригинальных тегов.
    Возвращает словарь с ключами 'colorspace', 'primaries', 'trc', 'range'.
    Значения - строки, понятные FFmpeg/PyAV.
    """
    cs_tag = original_tags.get('color_space')
    cp_tag = original_tags.get('color_primaries')
    ct_tag = original_tags.get('color_transfer')
    cr_tag = original_tags.get('color_range')

    # Значения по умолчанию (для HD)
    assumed_cs = 'bt709'
    assumed_cp = 'bt709'
    assumed_ct = 'bt709'
    assumed_cr = 'tv'

    if height <= 576:
        assumed_cs = 'bt470bg' if cs_tag and '601' not in cs_tag else 'bt470bg'  # или smpte170m
        assumed_cp = 'bt470bg' if cp_tag and '601' not in cp_tag else 'bt470bg'  # или smpte170m
        assumed_ct = 'gamma28' if ct_tag and '601' not in ct_tag else 'gamma28'  # или smpte170m
        assumed_cr = cr_tag if cr_tag in ['tv', 'pc', 'mpeg', 'jpeg'] else 'tv'
    elif height > 1080:  # Пример для UHD
        if cs_tag == 'bt2020ncl': assumed_cs = 'bt2020ncl'
        if cp_tag == 'bt2020': assumed_cp = 'bt2020'
        if ct_tag and ('bt2020' in ct_tag or 'pq' in ct_tag or 'hlg' in ct_tag):
            assumed_ct = ct_tag
        else:
            assumed_ct = 'bt709'  # Fallback к bt709 для SDR UHD

        if cr_tag == 'pc': assumed_cr = 'pc'
    else:  # HD
        if cs_tag and '709' in cs_tag: assumed_cs = cs_tag
        if cp_tag and '709' in cp_tag: assumed_cp = cp_tag
        if ct_tag and '709' in ct_tag: assumed_ct = ct_tag
        if cr_tag == 'pc': assumed_cr = 'pc'

    final_props = {
        'colorspace': assumed_cs,
        'primaries': assumed_cp,
        'trc': assumed_ct,
        'range': assumed_cr
    }
    logging.debug(f"Определены цветовые свойства для {width}x{height}: {final_props} (исходные: {original_tags})")
    return final_props


# @profile
def write_head_only(
        watermarked_head_frames: List[np.ndarray],
        all_audio_packets: Optional[List[av.Packet]],
        input_metadata: Dict[str, Any],
        temp_head_path: str,
        target_video_encoder_lib: str,
        video_encoder_options: Optional[Dict[str, str]] = None,
        audio_encoder_options: Optional[Dict[str, str]] = None
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """
    Записывает обработанные кадры "головы" и соответствующую им часть аудио
    во временный файл. Аудио всегда перекодируется в AAC.

    Возвращает кортеж:
        - float: Точная длительность записанной ВИДЕО части "головы" в секундах.
                 None в случае критической ошибки. 0.0, если кадров не было.
        - dict: Словарь с параметрами, использованными для кодирования "головы".
                None в случае критической ошибки.
                Ключи: 'video_encoder_lib', 'video_options', 'video_pix_fmt', 'video_fps_fraction', 'video_time_base',
                       'audio_codec' (всегда 'aac'), 'audio_options', 'audio_rate', 'audio_layout', 'audio_time_base'.
    """
    if not PYAV_AVAILABLE:
        logging.error("PyAV недоступен для write_head_only.")
        return None, None
    if not watermarked_head_frames:
        logging.warning("write_head_only: Нет кадров для записи в 'голову'.")
        return 0.0, {}

    num_head_frames = len(watermarked_head_frames)
    # Аудио для головы всегда AAC
    actual_audio_codec_for_head = 'aac'

    logging.info(f"Запись 'головы' в '{temp_head_path}' ({num_head_frames} кадров)...")
    logging.info(
        f"  Видеокодер для головы: {target_video_encoder_lib}, Аудио для головы: {actual_audio_codec_for_head}")

    output_container: Optional[av.container.Container] = None
    video_stream_out: Optional[av.stream.Stream] = None  # Переименовал для ясности
    audio_stream_out: Optional[av.stream.Stream] = None  # Переименовал для ясности
    input_audio_decoder: Optional[av.codec.context.CodecContext] = None
    video_reformatter: Optional[VideoReformatter] = None  # Переименовал

    actual_video_duration_sec: Optional[float] = None
    last_successfully_muxed_video_pts: int = -1

    last_encoded_video_frame_pts: int = -1
    video_frame_duration_in_tb: int = 0

    # --- Параметры кодирования, которые будут возвращены ---
    used_encoding_params: Dict[str, Any] = {
        'video_encoder_lib': target_video_encoder_lib,
        'video_options': video_encoder_options.copy() if video_encoder_options else {},
        'video_pix_fmt': 'yuv420p',
        'video_fps_fraction': None,
        'video_time_base': None,
        'audio_codec': actual_audio_codec_for_head,
        'audio_options': audio_encoder_options.copy() if audio_encoder_options else {},
        'audio_rate': None,
        'audio_layout': None,
        'audio_time_base': None,
    }

    try:
        width = input_metadata.get('width')
        height = input_metadata.get('height')
        fps_meta = input_metadata.get('fps')

        if not (width and height and width > 0 and height > 0):
            logging.error(f"Некорректные размеры кадра из input_metadata: {width}x{height}")
            return None, None

        fps_to_use_float = float(FPS)
        if fps_meta:
            try:
                fps_meta_float = float(fps_meta)
                if fps_meta_float > 0:
                    fps_to_use_float = fps_meta_float
                else:
                    logging.warning(
                        f"FPS из метаданных ({fps_meta}) невалиден (<=0). Используется fallback FPS={fps_to_use_float}.")
            except ValueError:
                logging.warning(
                    f"Не удалось преобразовать FPS из метаданных ('{fps_meta}') во float. Используется fallback FPS={fps_to_use_float}.")
        else:
            logging.warning(f"FPS не найден в метаданных. Используется fallback FPS={fps_to_use_float}.")

        used_encoding_params['video_fps_fraction'] = Fraction(fps_to_use_float).limit_denominator()

        has_audio_original = input_metadata.get('has_audio', False)
        process_audio = has_audio_original and (all_audio_packets is not None)

        output_container = av.open(temp_head_path, mode='w', metadata_errors='ignore')
        if output_container is None:
            raise av.FFmpegError(f"Не удалось открыть выходной контейнер (av.open вернул None) для '{temp_head_path}'")

        # --- Настройка видеопотока ---
        original_color_tags = {
            'color_space': input_metadata.get('color_space_tag'),
            'color_primaries': input_metadata.get('color_primaries_tag'),
            'color_transfer': input_metadata.get('color_transfer_tag'),
            'color_range': input_metadata.get('color_range_tag'),
        }
        color_props_for_output = get_assumed_color_properties(width, height, original_color_tags)

        if color_props_for_output.get('colorspace'): used_encoding_params['video_options']['colorspace'] = \
        color_props_for_output['colorspace']
        if color_props_for_output.get('primaries'): used_encoding_params['video_options']['color_primaries'] = \
        color_props_for_output['primaries']
        if color_props_for_output.get('trc'): used_encoding_params['video_options']['color_trc'] = \
        color_props_for_output['trc']
        if color_props_for_output.get('range'): used_encoding_params['video_options']['color_range'] = \
        color_props_for_output['range']

        video_stream_out = output_container.add_stream(target_video_encoder_lib,
                                                       rate=used_encoding_params['video_fps_fraction'])
        video_stream_out.width = width
        video_stream_out.height = height
        video_stream_out.pix_fmt = used_encoding_params['video_pix_fmt']  # 'yuv420p'

        if used_encoding_params['video_options']:
            video_stream_out.codec_context.options = used_encoding_params['video_options']
        logging.debug(f"Опции видеокодека для головы: {video_stream_out.codec_context.options}")

        # Установка time_base для видеопотока
        input_video_time_base = input_metadata.get('video_time_base')
        if input_video_time_base and isinstance(input_video_time_base,
                                                Fraction) and input_video_time_base.denominator != 0:
            video_stream_out.time_base = input_video_time_base
            logging.info(f"Видеопоток 'головы' time_base установлен из оригинала: {video_stream_out.time_base}")
        else:
            if video_stream_out.time_base is None or video_stream_out.time_base.denominator == 0:
                # Типичное значение для многих кодеров/контейнеров
                video_stream_out.time_base = Fraction(1, 90000)
            logging.info(f"Видеопоток 'головы' time_base (установлен PyAV или fallback): {video_stream_out.time_base}")
        used_encoding_params['video_time_base'] = video_stream_out.time_base

        if video_stream_out.time_base and used_encoding_params['video_fps_fraction'] and float(
                used_encoding_params['video_fps_fraction']) > 0:
            try:
                # duration = 1 / FPS.  duration_in_tb = duration / time_base_value
                video_frame_duration_in_tb = int(round(
                    (1.0 / float(used_encoding_params['video_fps_fraction'])) / float(video_stream_out.time_base)
                ))
            except (ZeroDivisionError, ValueError, TypeError) as e_calc_dur:
                logging.warning(f"Не удалось рассчитать video_frame_duration_in_tb: {e_calc_dur}. Используется 1.")
                video_frame_duration_in_tb = 1
        if video_frame_duration_in_tb <= 0:
            video_frame_duration_in_tb = 1
            logging.warning(
                f"Рассчитанная video_frame_duration_in_tb ({video_frame_duration_in_tb}) некорректна. Установлено в 1.")
        logging.debug(
            f"Расчетная длительность одного видеокадра для 'головы' (в video_time_base): {video_frame_duration_in_tb}")

        # --- Настройка аудиопотока---
        decoded_audio_frames_for_head: List[av.AudioFrame] = []
        # Оценка длительности головы в секундах для отсечки декодирования аудио
        head_video_duration_estimated_sec = num_head_frames / float(used_encoding_params['video_fps_fraction']) \
            if float(used_encoding_params['video_fps_fraction']) > 0 else 0.0

        if process_audio:
            input_audio_codec_name = input_metadata.get('audio_codec')
            input_audio_rate = input_metadata.get('audio_rate')
            input_audio_layout_str = input_metadata.get('audio_layout')
            input_audio_time_base = input_metadata.get('audio_time_base')

            if not (input_audio_codec_name and input_audio_rate and input_audio_layout_str and input_audio_time_base and
                    isinstance(input_audio_time_base, Fraction) and input_audio_time_base.denominator != 0):
                logging.warning(
                    "Аудио: Недостаточно валидных метаданных для обработки аудио в 'голове'. Аудио не будет добавлено.")
                process_audio = False
            else:
                logging.info(
                    f"Аудио для 'головы': Перекодирование из '{input_audio_codec_name}' в '{actual_audio_codec_for_head}', Rate={input_audio_rate}, Layout={input_audio_layout_str}")
                used_encoding_params['audio_rate'] = input_audio_rate
                used_encoding_params['audio_layout'] = input_audio_layout_str
                try:
                    audio_stream_out = output_container.add_stream(actual_audio_codec_for_head, rate=input_audio_rate)
                    audio_stream_out.codec_context.layout = input_audio_layout_str

                    if not used_encoding_params['audio_options']:
                        used_encoding_params['audio_options'] = {'b:a': '128k'}
                    audio_stream_out.codec_context.options = used_encoding_params['audio_options']
                    logging.debug(
                        f"Опции аудиокодера ({actual_audio_codec_for_head}) для головы: {audio_stream_out.codec_context.options}")

                    if audio_stream_out.time_base is None or audio_stream_out.time_base.denominator == 0:
                        audio_stream_out.time_base = Fraction(1, input_audio_rate)
                    logging.info(f"Аудиопоток 'головы' time_base: {audio_stream_out.time_base}")
                    used_encoding_params['audio_time_base'] = audio_stream_out.time_base

                    # Декодирование аудиопакетов из оригинала
                    input_audio_decoder = av.Codec(input_audio_codec_name, 'r').create()
                    if input_audio_decoder is None:
                        raise av.FFmpegError(f"Не удалось создать аудио декодер для '{input_audio_codec_name}'")

                    in_audio_ctx_params = input_metadata.get('audio_codec_context_params')
                    if in_audio_ctx_params:
                        if in_audio_ctx_params.get('format'): input_audio_decoder.format = in_audio_ctx_params['format']
                        if in_audio_ctx_params.get('layout'): input_audio_decoder.layout = in_audio_ctx_params['layout']
                        if in_audio_ctx_params.get('rate'): input_audio_decoder.sample_rate = in_audio_ctx_params[
                            'rate']
                        if in_audio_ctx_params.get('extradata'): input_audio_decoder.extradata = in_audio_ctx_params[
                            'extradata']

                    if input_audio_decoder.layout is None and input_audio_layout_str: input_audio_decoder.layout = input_audio_layout_str
                    if input_audio_decoder.sample_rate is None and input_audio_rate: input_audio_decoder.sample_rate = input_audio_rate

                    logging.debug(
                        f"Декодирование аудиопакетов для 'головы' (оценочная длина видео ~{head_video_duration_estimated_sec:.3f}s)")
                    processed_audio_packet_count = 0
                    for audio_packet_original in all_audio_packets or []:
                        processed_audio_packet_count += 1
                        if audio_packet_original.dts is None: continue

                        packet_time_sec = 0.0
                        if audio_packet_original.pts is not None:
                            try:
                                packet_time_sec = float(audio_packet_original.pts * input_audio_time_base)
                            except Exception:
                                pass

                        if packet_time_sec > head_video_duration_estimated_sec + 2.0:
                            logging.debug(
                                f"Аудиопакет {processed_audio_packet_count} PTS {audio_packet_original.pts} ({packet_time_sec:.3f}s) "
                                f"за пределами ОЦЕНОЧНОЙ длины видео 'головы' + буфер. Прерывание декодирования аудио.")
                            break
                        try:
                            decoded_frames = input_audio_decoder.decode(audio_packet_original)
                            if decoded_frames: decoded_audio_frames_for_head.extend(decoded_frames)
                        except av.FFmpegError as e_decode_audio:
                            if e_decode_audio.errno == -11 or 'again' in str(e_decode_audio).lower():
                                logging.debug(
                                    f"Ошибка декодирования аудиопакета {processed_audio_packet_count} (EAGAIN), требуется больше данных.")
                                continue
                            logging.warning(
                                f"Ошибка декодирования аудиопакета {processed_audio_packet_count} из оригинала: {e_decode_audio}")
                    try:  # Flush аудио декодера
                        decoded_frames_flush = input_audio_decoder.decode(None)
                        if decoded_frames_flush: decoded_audio_frames_for_head.extend(decoded_frames_flush)
                    except av.FFmpegError:
                        pass
                    logging.info(
                        f"Для 'головы' декодировано {len(decoded_audio_frames_for_head)} аудиокадров из оригинала.")

                except Exception as e_setup_audio_stream:
                    logging.error(f"Ошибка при настройке или декодировании аудио для 'головы': {e_setup_audio_stream}",
                                  exc_info=True)
                    process_audio = False
                    audio_stream_out = None

        # --- Кодирование и мультиплексирование видео ---
        logging.info(f"Кодирование и мультиплексирование {num_head_frames} видеокадров для 'головы'...")
        encoded_video_frame_count = 0
        current_video_pts_for_frame: int = 0

        for frame_idx, bgr_frame_np in enumerate(watermarked_head_frames):
            try:
                if not isinstance(bgr_frame_np, np.ndarray) or bgr_frame_np.shape[:2] != (height, width):
                    logging.warning(f"Пропуск некорректного видеокадра (индекс {frame_idx}) при записи 'головы'.")
                    continue

                video_frame_in = av.VideoFrame.from_ndarray(bgr_frame_np, format='bgr24')

                if video_reformatter is None:
                    video_reformatter = VideoReformatter(
                        video_frame_in.width, video_frame_in.height, used_encoding_params['video_pix_fmt'],  # 'yuv420p'
                        src_format='bgr24',
                        src_colorspace=None,
                        dst_colorspace=color_props_for_output.get('colorspace'),
                        dst_primaries=color_props_for_output.get('primaries'),
                        dst_trc=color_props_for_output.get('trc'),
                        dst_color_range=color_props_for_output.get('range')
                    )
                video_frame_to_encode = video_reformatter.reformat(video_frame_in)
                video_frame_to_encode.pts = current_video_pts_for_frame

                encoded_video_packets = video_stream_out.encode(video_frame_to_encode)

                if encoded_video_packets:
                    last_encoded_video_frame_pts = current_video_pts_for_frame
                    for packet_vid in encoded_video_packets:
                        output_container.mux(packet_vid)
                    encoded_video_frame_count += 1

                current_video_pts_for_frame += video_frame_duration_in_tb

            except Exception as e_encode_video_frame:
                logging.error(
                    f"Ошибка при обработке/кодировании видеокадра {frame_idx} для 'головы': {e_encode_video_frame}",
                    exc_info=True)
                raise

        # Flush видеокодера
        logging.debug("Завершение (flush) видеокодера для 'головы'...")
        encoded_video_packets_flush = video_stream_out.encode(None)
        if encoded_video_packets_flush:
            output_container.mux(encoded_video_packets_flush)
        logging.info(f"Закодировано и записано {encoded_video_frame_count} видеокадров в 'голову'.")

        # Точный расчет длительности видео на основе PTS последнего успешно закодированного кадра и его длительности
        if encoded_video_frame_count > 0 and last_encoded_video_frame_pts >= 0 and video_frame_duration_in_tb > 0 and video_stream_out.time_base:
            try:
                # Конечный PTS видеопотока = PTS последнего кадра + его длительность
                effective_last_video_pts = last_encoded_video_frame_pts + video_frame_duration_in_tb
                duration_value_sec = float(effective_last_video_pts * video_stream_out.time_base)
                if duration_value_sec >= 0:
                    actual_video_duration_sec = duration_value_sec
                else:
                    logging.error(f"Рассчитана отрицательная видео длительность ({duration_value_sec}s) для 'головы'.")
                    actual_video_duration_sec = None
            except Exception as e_calc_final_duration:
                logging.error(f"Ошибка при финальном расчете видео длительности 'головы': {e_calc_final_duration}")
                actual_video_duration_sec = None
        elif encoded_video_frame_count == 0 and num_head_frames > 0:
            actual_video_duration_sec = None
            logging.error("Ни один видеокадр не был успешно закодирован для 'головы', хотя кадры для обработки были.")
        else:
            actual_video_duration_sec = 0.0

        if actual_video_duration_sec is not None:
            logging.info(f"Финальная ТОЧНАЯ ВИДЕО длительность 'головы': {actual_video_duration_sec:.9f}s")
        else:
            logging.error("Не удалось рассчитать точную видео длительность 'головы'.")

        # --- Кодирование и мультиплексирование аудио (если оно есть) ---
        if process_audio and audio_stream_out and decoded_audio_frames_for_head:
            duration_limit_for_audio_sec = actual_video_duration_sec if (
                        actual_video_duration_sec is not None and actual_video_duration_sec > 0) \
                else head_video_duration_estimated_sec

            logging.info(
                f"Кодирование и мультиплексирование {len(decoded_audio_frames_for_head)} аудиокадров для 'головы' "
                f"(до ~{duration_limit_for_audio_sec:.3f}s видеовремени)..."
            )
            encoded_audio_frame_count = 0
            for audio_frame_idx, audio_frame_to_encode in enumerate(decoded_audio_frames_for_head):
                try:
                    audio_frame_pts_sec = 0.0
                    if audio_frame_to_encode.pts is not None and audio_stream_out.time_base:
                        try:
                            if input_audio_time_base and input_audio_time_base.denominator != 0:
                                audio_frame_pts_sec = float(audio_frame_to_encode.pts * input_audio_time_base)
                            else:
                                pass
                        except Exception:
                            pass

                    if audio_frame_pts_sec > duration_limit_for_audio_sec + 0.02:  # +20мс буфер
                        logging.debug(
                            f"Пропуск аудиокадра {audio_frame_idx} для 'головы' (его PTS {audio_frame_pts_sec:.3f}s "
                            f"> лимита {duration_limit_for_audio_sec:.3f}s + буфер)")
                        continue

                    encoded_audio_packets = audio_stream_out.encode(audio_frame_to_encode)
                    if encoded_audio_packets:
                        encoded_audio_frame_count += 1
                        for packet_aud in encoded_audio_packets:
                            output_container.mux(packet_aud)
                except Exception as e_encode_audio_frame:
                    logging.warning(
                        f"Ошибка кодирования/мультиплексирования аудиокадра {audio_frame_idx} для 'головы': {e_encode_audio_frame}")

            # Flush аудиокодера
            logging.debug("Завершение (flush) аудиокодера для 'головы'...")
            encoded_audio_packets_flush = audio_stream_out.encode(None)
            if encoded_audio_packets_flush:
                output_container.mux(encoded_audio_packets_flush)
            logging.info(f"Закодировано и записано {encoded_audio_frame_count} аудиокадров в 'голову'.")

        logging.info(f"Запись 'головы' в '{temp_head_path}' завершена.")
        return actual_video_duration_sec, used_encoding_params

    except av.FFmpegError as e_ffmpeg:
        logging.error(f"Критическая ошибка PyAV/FFmpeg при записи 'головы' в '{temp_head_path}': {e_ffmpeg}",
                      exc_info=True)
        return None, None
    except Exception as e_general:
        logging.error(f"Неожиданная критическая ошибка при записи 'головы' в '{temp_head_path}': {e_general}",
                      exc_info=True)
        return None, None
    finally:
        if output_container:
            try:
                output_container.close()
            except av.FFmpegError as e_close_container:
                logging.error(f"Ошибка при закрытии контейнера 'головы' ('{temp_head_path}'): {e_close_container}")


def concatenate_smart_stitch(
        original_input_path: str,
        temp_head_path: str,
        final_output_path: str,
        head_end_time_sec: float,
        input_metadata: Dict[str, Any],
        iframe_times_sec: List[float],
        head_encoding_params: Dict[str, Any],
        gap_threshold_sec: float = 0.005
) -> bool:
    """
    Выполняет "умную" склейку видео ("Голова + Переход + Хвост_Копия"):
    1. Анализирует "дыру" между концом головы и следующим I-кадром оригинала.
    2. Если "дыра" больше `gap_threshold_sec`, создает ПЕРЕКОДИРОВАННЫЙ
       "переходный" сегмент (`temp_transition`) для ее заполнения, используя
       параметры кодирования головы.
    3. Создает КОПИРОВАННЫЙ "хвост" (`temp_tail_copy`), начиная с подходящего
       I-кадра оригинала (после головы или после перехода).
    4. Пытается "очистить" копированный хвост перепаковкой (`temp_tail_copy_clean`).
    5. Склеивает все сегменты (голова, [переход], хвост/очищенный хвост) с помощью
       FFmpeg concat демультиплексора и опции `-c copy` для ВСЕХ потоков.

    Args:
        original_input_path: Путь к оригинальному видеофайлу.
        temp_head_path: Путь к временному файлу с обработанной "головой".
        final_output_path: Путь для сохранения итогового склеенного видео.
        head_end_time_sec: Точная длительность видеоданных в `temp_head_path` в секундах.
        input_metadata: Словарь с метаданными оригинального видео.
        iframe_times_sec: Отсортированный список времен начала I-кадров оригинала в секундах.
        head_encoding_params: Словарь с параметрами кодирования головы.
        gap_threshold_sec: Минимальная длительность "дыры" в секундах.

    Returns:
        True в случае успеха, False в случае ошибки.
    """
    process_start_time = time.time()
    logging.info(f"Запуск concatenate_smart_stitch: Голова='{temp_head_path}', Выход='{final_output_path}'")
    logging.info(f"  Конец головы: {head_end_time_sec:.9f}s. Порог для переходного сегмента: {gap_threshold_sec}s.")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logging.error("FFmpeg не найден.")
        return False

    required_head_params = [
        'video_encoder_lib', 'video_options', 'video_pix_fmt', 'video_fps_fraction',
        'video_time_base', 'audio_codec', 'audio_options', 'audio_rate',
        'audio_layout', 'audio_time_base']
    if not head_encoding_params or not all(key in head_encoding_params for key in required_head_params):
        logging.error("Недостаточно параметров кодирования головы.")
        logging.debug(f"Полученные параметры головы: {head_encoding_params}")
        return False

    temp_transition_path: Optional[str] = None
    temp_tail_copy_path: Optional[str] = None
    temp_tail_copy_clean_path: Optional[str] = None
    list_file_path: Optional[str] = None
    files_to_concat: List[str] = [temp_head_path]

    try:
        search_start_time = head_end_time_sec - 0.001
        if iframe_times_sec is None:
            logging.error("Список времен I-кадров не предоставлен.")
            return False
        suitable_iframe_times = [t for t in iframe_times_sec if t >= search_start_time]

        if not suitable_iframe_times:
            logging.warning(f"Не найден I-кадр после {search_start_time:.6f}s. Голова покрывает все видео.")
            try:
                if os.path.exists(final_output_path) and final_output_path != temp_head_path:
                    time.sleep(0.1);
                    os.remove(final_output_path)
                if final_output_path != temp_head_path:
                    time.sleep(0.1);
                    shutil.copy2(temp_head_path, final_output_path)
                    time.sleep(0.1);
                    os.remove(temp_head_path)
                    logging.info(f"Голова скопирована в '{final_output_path}', исходник удален.")
                else:
                    logging.info(f"Голова уже финальный файл.")
                return True
            except Exception as e_copyhead:
                logging.error(f"Ошибка при копировании/удалении головы: {e_copyhead}", exc_info=True)
                return False

        keyframe_tail_start_sec = suitable_iframe_times[0]
        gap_duration = keyframe_tail_start_sec - head_end_time_sec
        logging.info(
            f"Точка стыка: Конец головы={head_end_time_sec:.9f}s, Начало след. I-кадра={keyframe_tail_start_sec:.9f}s. Дыра={gap_duration:.9f}s")

        output_dir = os.path.dirname(temp_head_path)
        base_name_head = os.path.splitext(os.path.basename(temp_head_path))[0].replace('_head', '')
        output_extension = os.path.splitext(final_output_path)[1]
        try:
            current_pid = os.getpid()
        except Exception:
            current_pid = "nopid"

        temp_transition_path = os.path.join(output_dir,
                                            f"{base_name_head}_transition_pid{current_pid}{output_extension}")
        temp_tail_copy_path = os.path.join(output_dir, f"{base_name_head}_tail_copy_pid{current_pid}{output_extension}")
        temp_tail_copy_clean_path = temp_tail_copy_path.replace('_tail_copy_',
                                                                '_tail_clean_')  # Имя для очищенного хвоста

        create_transition = gap_duration > gap_threshold_sec
        transition_created = False
        tail_copy_created = False
        tail_copy_cleaned = False

        # --- Этап 1: Создание Переходного Сегмента ---
        if create_transition:
            logging.info(
                f"Создание ПЕРЕКОДИРОВАННОГО переходного сегмента '{temp_transition_path}' (длительность ~{gap_duration:.3f}s)...")
            transition_start_sec = head_end_time_sec
            transition_duration_sec = max(0.0, keyframe_tail_start_sec - transition_start_sec)
            seek_transition_start = max(0.0, transition_start_sec - 1.0)
            trim_start_relative = max(0.0, transition_start_sec - seek_transition_start)

            cmd_create_transition = [
                ffmpeg_path, '-y',
                '-ss', f"{seek_transition_start:.9f}",
                '-i', original_input_path,
                '-vf',
                f"trim=start={trim_start_relative:.9f}:duration={transition_duration_sec:.9f},setpts=PTS-STARTPTS",
                '-af',
                f"atrim=start={trim_start_relative:.9f}:duration={transition_duration_sec:.9f},asetpts=PTS-STARTPTS",
                '-c:v', head_encoding_params['video_encoder_lib'],
                '-pix_fmt', head_encoding_params['video_pix_fmt'],
                '-r', str(float(head_encoding_params['video_fps_fraction'])),
                '-video_track_timescale', str(head_encoding_params['video_time_base'].denominator)]

            video_opts = head_encoding_params.get('video_options', {})
            if isinstance(video_opts, dict):
                for key, value in video_opts.items():
                    try:
                        cmd_create_transition.extend([f'-{str(key)}', str(value)])
                    except Exception as e_vopt_str:
                        logging.warning(f"Не удалось добавить видео опцию {key}={value}: {e_vopt_str}")
            else:
                logging.warning(f"video_options в head_encoding_params не являются словарем: {video_opts}")

            audio_layout_str = head_encoding_params.get('audio_layout', 'stereo')
            audio_rate_str = str(head_encoding_params.get('audio_rate', '44100'))
            audio_codec_str = head_encoding_params.get('audio_codec', 'aac')
            audio_opts = head_encoding_params.get('audio_options', {})
            if not isinstance(audio_layout_str, str) or not audio_layout_str:
                logging.warning(f"Некорректный audio_layout ('{audio_layout_str}'). Fallback='stereo'.")
                audio_layout_str = 'stereo'

            cmd_create_transition_audio = ['-c:a', audio_codec_str, '-ar', audio_rate_str, '-channel_layout',
                                           audio_layout_str]
            logging.debug(
                f"Установка аудио параметров для перехода: codec={audio_codec_str}, rate={audio_rate_str}, layout={audio_layout_str}")

            if isinstance(audio_opts, dict):
                for key, value in audio_opts.items():
                    if key.lower() not in ['ac', 'channel_layout', 'layout']:
                        try:
                            cmd_create_transition_audio.extend([f'-{str(key)}', str(value)])
                        except Exception as e_aopt_str:
                            logging.warning(f"Не удалось добавить аудио опцию {key}={value}: {e_aopt_str}")
            else:
                logging.warning(f"audio_options в head_encoding_params не являются словарем: {audio_opts}")
            cmd_create_transition.extend(cmd_create_transition_audio)

            cmd_create_transition.extend(
                ['-force_key_frames', 'expr:eq(n,0)', '-map_metadata', '-1', temp_transition_path])

            logging.debug(f"Команда FFmpeg для создания переходного сегмента: {' '.join(cmd_create_transition)}")
            try:
                result_transition = subprocess.run(cmd_create_transition, check=False, capture_output=True, text=True,
                                                   encoding='utf-8', errors='replace')
                if result_transition.returncode == 0 and os.path.exists(temp_transition_path) and os.path.getsize(
                        temp_transition_path) > 100:
                    logging.info(f"Переходный сегмент '{temp_transition_path}' успешно создан.")
                    transition_created = True
                    files_to_concat.append(temp_transition_path)
                else:
                    logging.error(
                        f"Ошибка FFmpeg при создании перехода (код {result_transition.returncode}).\nStderr: {result_transition.stderr}")
            except Exception as e_transition_run:
                logging.error(f"Ошибка создания перехода: {e_transition_run}", exc_info=True)

            # Если переход создать не удалось, но он был нужен - это ошибка для smart stitch
            if create_transition and not transition_created:
                logging.error("Не удалось создать необходимый переходный сегмент. Отмена операции.")
                return False

                # --- Этап 2: Создание КОПИРОВАННОГО Хвоста (`temp_tail_copy.mp4`) ---
        start_copy_sec = keyframe_tail_start_sec
        logging.info(f"Создание КОПИРОВАННОГО хвоста '{temp_tail_copy_path}', начиная с {start_copy_sec:.9f}s...")

        original_audio_codec = input_metadata.get('audio_codec')
        output_ext_tail = os.path.splitext(temp_tail_copy_path)[1].lower()
        can_copy_tail_audio = (original_audio_codec == 'aac') and (output_ext_tail in ['.mp4', '.mov', '.mkv'])
        tail_copy_audio_action = 'copy' if can_copy_tail_audio else 'aac'
        logging.info(f"  Действие для аудио при создании копированного хвоста: {tail_copy_audio_action}")

        cmd_create_tail_copy = [
            ffmpeg_path, '-y',
            '-ss', f"{start_copy_sec:.9f}",
            '-i', original_input_path,
            '-c:v', 'copy',
            '-c:a', tail_copy_audio_action,
            '-map_metadata', '-1',
            temp_tail_copy_path
        ]
        if tail_copy_audio_action != 'copy':
            original_audio_bitrate_val = input_metadata.get('audio_bitrate')
            bitrate_str_tail = str(
                original_audio_bitrate_val) if original_audio_bitrate_val and original_audio_bitrate_val >= 32000 else "192k"
            cmd_create_tail_copy.insert(-1, '-b:a');
            cmd_create_tail_copy.insert(-1, bitrate_str_tail)
            logging.info(
                f"    (Перекодирование аудио хвоста в {tail_copy_audio_action} с битрейтом {bitrate_str_tail})")

        logging.debug(f"Команда FFmpeg для создания копированного хвоста: {' '.join(cmd_create_tail_copy)}")
        try:
            result_tail_copy = subprocess.run(cmd_create_tail_copy, check=False, capture_output=True, text=True,
                                              encoding='utf-8', errors='replace')
            if result_tail_copy.returncode == 0 and os.path.exists(temp_tail_copy_path) and os.path.getsize(
                    temp_tail_copy_path) > 1024:
                logging.info(f"Копированный хвост '{temp_tail_copy_path}' успешно создан.")
                tail_copy_created = True
                # Не добавляем в files_to_concat напрямую, сначала попробуем очистить
            else:
                logging.error(
                    f"Ошибка FFmpeg при создании копированного хвоста (код {result_tail_copy.returncode}).\nStderr: {result_tail_copy.stderr}")
        except Exception as e_tail_copy:
            logging.error(f"Ошибка создания копированного хвоста: {e_tail_copy}", exc_info=True)

        if not tail_copy_created:
            logging.error("Не удалось создать копированный хвост. Отмена операции.")
            return False  # Критическая ошибка

        # --- Этап 2.5: "Очистка" Копированного Хвоста Перепаковкой ---
        path_to_concat_for_tail = temp_tail_copy_path
        if tail_copy_created:
            logging.info(f"Этап 2.5: Очистка/Перепаковка копированного хвоста в '{temp_tail_copy_clean_path}'...")
            cmd_clean_tail = [
                ffmpeg_path, '-y',
                '-i', temp_tail_copy_path,
                '-c', 'copy',
                '-map', '0',
                '-movflags', '+faststart',
                temp_tail_copy_clean_path
            ]
            logging.debug(f"Команда FFmpeg для очистки хвоста: {' '.join(cmd_clean_tail)}")
            try:
                result_clean_tail = subprocess.run(cmd_clean_tail, check=False, capture_output=True, text=True,
                                                   encoding='utf-8', errors='replace')
                if result_clean_tail.returncode == 0 and os.path.exists(temp_tail_copy_clean_path) and os.path.getsize(
                        temp_tail_copy_clean_path) > 100:
                    logging.info(f"Очищенный хвост '{temp_tail_copy_clean_path}' успешно создан.")
                    tail_copy_cleaned = True
                    path_to_concat_for_tail = temp_tail_copy_clean_path
                else:
                    logging.warning(
                        f"Не удалось очистить хвост (код {result_clean_tail.returncode}). Будет использован неочищенный.\nStderr: {result_clean_tail.stderr}")
                    # path_to_concat_for_tail остается temp_tail_copy_path
            except Exception as e_clean_tail:
                logging.warning(
                    f"Не удалось очистить хвост из-за исключения: {e_clean_tail}. Будет использован неочищенный.")
                # path_to_concat_for_tail остается temp_tail_copy_path
        files_to_concat.append(path_to_concat_for_tail)

        # --- Этап 3: Финальная Склейка (concat демультиплексор с -c copy) ---
        logging.info(
            f"Этап 3: Финальная склейка {len(files_to_concat)} сегментов в '{final_output_path}' (с -c copy)...")
        success_concat = False
        try:
            list_file_content = ""
            for f_path in files_to_concat:
                abs_path = os.path.abspath(f_path).replace('\\', '/')
                list_file_content += f"file '{abs_path}'\n"

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as list_file_obj:
                list_file_path = list_file_obj.name
                list_file_obj.write(list_file_content)
            logging.debug(f"Создан временный файл списка для concat: '{list_file_path}'")

            # Финальная склейка с -c copy для ВСЕХ потоков
            cmd_concat_final = [
                ffmpeg_path, '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file_path,
                '-c', 'copy',
                '-movflags', '+faststart',
                final_output_path
            ]
            logging.debug(f"Команда финальной склейки (-c copy): {' '.join(cmd_concat_final)}")

            result_concat = subprocess.run(cmd_concat_final, check=False, capture_output=True, text=True,
                                           encoding='utf-8', errors='replace')

            if result_concat.returncode == 0 and os.path.exists(final_output_path) and os.path.getsize(
                    final_output_path) > 1024:
                logging.info(f"FFmpeg: Финальная склейка (-c copy) успешно завершена.")
                if result_concat.stderr: logging.debug(
                    f"  FFmpeg stderr (финальная склейка, успех):\n{result_concat.stderr}")
                success_concat = True
            else:
                logging.error(
                    f"Ошибка FFmpeg при финальной склейке (-c copy) (код {result_concat.returncode}).\nStderr: {result_concat.stderr}")

        except Exception as e_concat_final:
            logging.error(f"Неожиданная ошибка при финальной склейке: {e_concat_final}", exc_info=True)

        if success_concat:
            logging.info(
                f"Процесс concatenate_smart_stitch успешно завершен за {time.time() - process_start_time:.2f} сек.")
        else:
            logging.error(
                f"Процесс concatenate_smart_stitch завершился с ошибкой за {time.time() - process_start_time:.2f} сек.")

        return success_concat

    finally:
        files_to_delete = [list_file_path, temp_transition_path, temp_tail_copy_path, temp_tail_copy_clean_path]
        for f_path in files_to_delete:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    logging.debug(f"Удален временный файл: {f_path}")
                except OSError as e_cleanup:
                    logging.warning(f"Не удалось удалить временный файл '{f_path}': {e_cleanup}")


@profile
def embed_frame_pair(
        frame1_bgr: np.ndarray, frame2_bgr: np.ndarray, bits: List[int],
        selected_ring_indices: List[int], n_rings: int, frame_number: int,
        use_perceptual_masking: bool, embed_component: int,
        device: torch.device,
        dtcwt_fwd: 'DTCWTForward',  # Указываем тип для подсказок, если DTCWTForward определен
        dtcwt_inv: 'DTCWTInverse'  # Указываем тип для подсказок, если DTCWTInverse определен
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Встраивает биты (PyTorch DCT/SVD), используя GPU для адаптивной альфы.
    Логика применения дельты соответствует СТАБИЛЬНОЙ ВЕРСИИ (+=).
    """
    pair_index = frame_number // 2
    prefix_base = f"[P:{pair_index}]"

    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        logging.error(f"{prefix_base} Отсутствуют PyTorch Wavelets или DCT!")
        return None, None

    min_len_bits_rings = min(len(bits), len(selected_ring_indices))
    if len(bits) != len(selected_ring_indices):
        logging.warning(
            f"{prefix_base} Mismatch bits/rings: {len(bits)} vs {len(selected_ring_indices)}. Using min len {min_len_bits_rings}.")

    if min_len_bits_rings == 0:  # Если нет бит для встраивания или колец для использования
        logging.debug(f"{prefix_base} No bits/rings to process. Returning original frames.")
        return frame1_bgr, frame2_bgr  # Возвращаем исходные кадры

    bits_to_embed = bits[:min_len_bits_rings]
    rings_to_process = selected_ring_indices[:min_len_bits_rings]

    logging.debug(f"{prefix_base} --- Starting Embedding Pair (PyTorch GPU Alpha) for {len(bits_to_embed)} bits ---")
    try:
        # 1. Преобразование в Тензоры YCrCb и извлечение компонента
        f1_ycrcb_np = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2YCrCb)
        f2_ycrcb_np = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2YCrCb)

        # Нормализуем компонент [0, 255] в [0, 1] и переводим в тензор
        comp1_np_float = f1_ycrcb_np[:, :, embed_component].astype(np.float32) / 255.0
        comp2_np_float = f2_ycrcb_np[:, :, embed_component].astype(np.float32) / 255.0

        comp1_tensor = torch.from_numpy(comp1_np_float).to(device=device)
        comp2_tensor = torch.from_numpy(comp2_np_float).to(device=device)
        target_shape_hw = (frame1_bgr.shape[0], frame1_bgr.shape[1])

        # 2. Прямое DTCWT
        Yl_t, Yh_t = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, frame_number)
        Yl_t1, Yh_t1 = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, frame_number + 1)

        if Yl_t is None or Yh_t is None or Yl_t1 is None or Yh_t1 is None:
            logging.error(f"{prefix_base} DTCWT forward failed.")
            return None, None

        # Убираем batch/channel измерения, если они есть (должны быть (1,1,H,W) -> (H,W))
        if Yl_t.dim() > 2: Yl_t = Yl_t.squeeze(0).squeeze(0)
        if Yl_t1.dim() > 2: Yl_t1 = Yl_t1.squeeze(0).squeeze(0)

        # 3. Подготовка к модификации
        # ring_coords_t и ring_coords_t1 теперь должны быть списками тензоров координат,
        # полученными из GPU-версии ring_division
        ring_coords_t = ring_division(Yl_t, n_rings, frame_number)
        ring_coords_t1 = ring_division(Yl_t1, n_rings, frame_number + 1)

        if ring_coords_t is None or ring_coords_t1 is None:
            logging.error(f"{prefix_base} Ring division returned None.")
            return None, None

        if use_perceptual_masking:
            perceptual_mask_tensor = calculate_perceptual_mask_torch(
                comp1_tensor,  # Передаем исходный компонент (до DTCWT) для маски
                device=device,
                fn=frame_number
                # Ядра будут создаваться внутри calculate_perceptual_mask_torch
            )
            # calculate_perceptual_mask_torch должна вернуть тензор, который уже будет
            # иметь ту же форму, что и comp1_tensor (H,W).
            # Нам нужна маска с размерами Yl_t (LL-поддиапазона).
            # Если Yl_t имеет другие размеры, чем comp1_tensor, маску нужно ресайзить.
            if perceptual_mask_tensor.shape != Yl_t.shape:
                logging.warning(
                    f"{prefix_base} Perceptual mask shape {perceptual_mask_tensor.shape} (from comp1_tensor) != Yl_t shape {Yl_t.shape}. Resizing mask.")
                try:
                    perceptual_mask_tensor = F.interpolate(
                        perceptual_mask_tensor.unsqueeze(0).unsqueeze(0),  # (H,W) -> (1,1,H,W)
                        size=Yl_t.shape,  # Целевой размер (H_yl, W_yl)
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)  # (1,1,H_yl,W_yl) -> (H_yl,W_yl)
                except Exception as e_interp_mask:
                    logging.error(
                        f"{prefix_base} Failed to interpolate perceptual mask to Yl_t shape: {e_interp_mask}. Using ones mask.")
                    perceptual_mask_tensor = torch.ones_like(Yl_t, device=device)
        else:
            perceptual_mask_tensor = torch.ones_like(Yl_t, device=device)  # Единичная маска, если не используется

        modifications_count = 0
        Yl_t_mod = Yl_t.clone()
        Yl_t1_mod = Yl_t1.clone()

        # --- Цикл по выбранным кольцам для встраивания бит ---
        logging.debug(f"{prefix_base} --- Start Ring Loop (GPU Alpha, Embedding {len(bits_to_embed)} bits) ---")
        for ring_idx, bit_to_embed in zip(rings_to_process, bits_to_embed):
            prefix = f"[P:{pair_index} R:{ring_idx}]"
            logging.debug(f"{prefix} ------- Processing bit {bit_to_embed} -------")

            # Проверка валидности индекса кольца и наличия координат
            if not (0 <= ring_idx < n_rings and \
                    ring_idx < len(ring_coords_t) and ring_coords_t[ring_idx] is not None and \
                    ring_idx < len(ring_coords_t1) and ring_coords_t1[ring_idx] is not None):
                logging.warning(f"{prefix} Invalid ring_idx {ring_idx} or missing coordinates. Skipping.")
                continue

            coords1_tensor = ring_coords_t[ring_idx]
            coords2_tensor = ring_coords_t1[ring_idx]

            if coords1_tensor.shape[0] < 10 or coords2_tensor.shape[0] < 10:  # Минимальное количество пикселей в кольце
                logging.debug(
                    f"{prefix} Ring too small (coords1: {coords1_tensor.shape[0]}, coords2: {coords2_tensor.shape[0]}). Skipping.")
                continue

            try:
                # Извлечение координат и значений пикселей из тензоров
                rows1, cols1 = coords1_tensor[:, 0], coords1_tensor[:, 1]
                rows2, cols2 = coords2_tensor[:, 0], coords2_tensor[:, 1]

                # v1_tensor и v2_tensor - это тензоры значений пикселей из КОПИЙ Yl_t_mod и Yl_t1_mod
                # Используем .float() для гарантии типа перед DCT/SVD
                v1_tensor = Yl_t_mod[rows1, cols1].float()
                v2_tensor = Yl_t1_mod[rows2, cols2].float()

                min_s = min(v1_tensor.numel(), v2_tensor.numel())
                if min_s == 0:
                    logging.debug(f"{prefix} Empty ring after coordinate extraction for v1 or v2. Skipping.");
                    continue

                # Если размеры векторов пикселей не совпадают (маловероятно с GPU ring_division, но для безопасности)
                if v1_tensor.numel() != v2_tensor.numel():
                    v1_tensor = v1_tensor[:min_s]
                    v2_tensor = v2_tensor[:min_s]
                    # Важно: если обрезали, нужно также обновить rows1, cols1, rows2, cols2 для корректной записи обратно
                    rows1_actual, cols1_actual = rows1[:min_s], cols1[:min_s]
                    rows2_actual, cols2_actual = rows2[:min_s], cols2[:min_s]
                else:
                    rows1_actual, cols1_actual = rows1, cols1
                    rows2_actual, cols2_actual = rows2, cols2

                # Вычисляем адаптивную альфу с помощью новой GPU-версии функции
                # v1_tensor (значения из первого кадра пары) используется для вычисления альфы
                alpha_t = compute_adaptive_alpha_entropy_torch(v1_tensor, ring_idx,
                                                               frame_number)  # alpha_t теперь тензор

                # inv_a также будет тензором
                inv_a = torch.tensor(1.0, device=device, dtype=alpha_t.dtype) / (alpha_t + 1e-12)
                logging.debug(f"{prefix} PyTorch Adaptive alpha={alpha_t.item():.4f}")

                # DCT
                d1_tensor = dct1d_torch(v1_tensor)
                d2_tensor = dct1d_torch(v2_tensor)
                if not torch.isfinite(d1_tensor).all() or not torch.isfinite(d2_tensor).all():
                    logging.warning(f"{prefix} Non-finite DCT result. Skipping.");
                    continue

                # SVD (U, S, Vh.T)
                U1, S1_vec, Vh1_t = svd_torch(d1_tensor)
                U2, S2_vec, Vh2_t = svd_torch(d2_tensor)

                if U1 is None or S1_vec is None or Vh1_t is None or \
                        U2 is None or S2_vec is None or Vh2_t is None:
                    logging.warning(f"{prefix} SVD failed (returned None). Skipping.");
                    continue
                if S1_vec.numel() == 0 or S2_vec.numel() == 0:  # Проверка, что сингулярные числа не пусты
                    logging.warning(f"{prefix} Empty singular values vector from SVD. Skipping.");
                    continue

                s1 = S1_vec[0]  # s1, s2 - скалярные тензоры (главные сингулярные числа)
                s2 = S2_vec[0]
                if not torch.isfinite(s1) or not torch.isfinite(s2):  # Проверка на NaN/Inf
                    logging.warning(
                        f"{prefix} Non-finite singular values s1 ({s1.item()}) or s2 ({s2.item()}). Skipping.");
                    continue

                eps = torch.tensor(1e-12, device=device, dtype=s1.dtype)  # Эпсилон для защиты от деления на ноль
                s2_safe = s2 + eps if torch.abs(s2) < eps else s2  # s2_safe никогда не будет 0
                original_ratio = s1 / s2_safe  # original_ratio теперь тензор

                ns1, ns2 = s1.clone(), s2.clone()  # Клонируем тензоры s1, s2
                modified = False

                one_tensor = torch.tensor(1.0, device=device, dtype=s1.dtype)
                action_log_message = "No change needed"

                # Определяем текущий встроенный бит (0 или 1) на основе original_ratio
                current_bit_tensor = torch.where(original_ratio >= one_tensor,
                                                 torch.tensor(0, device=device, dtype=torch.long),
                                                 torch.tensor(1, device=device, dtype=torch.long))

                bit_to_embed_tensor = torch.tensor(bit_to_embed, device=device, dtype=torch.long)

                modify_needed_tensor = (current_bit_tensor != bit_to_embed_tensor)
                strengthen_needed_tensor = torch.tensor(False, device=device)  # Инициализируем как False тензор
                target_ratio_tensor = original_ratio.clone()

                if not modify_needed_tensor.item():  # Если .item() == False, т.е. биты совпадают
                    # Проверяем, нужно ли усиление
                    condition_strengthen0 = (bit_to_embed_tensor == 0) & (original_ratio < alpha_t)
                    condition_strengthen1 = (bit_to_embed_tensor == 1) & (
                                original_ratio > inv_a)
                    if condition_strengthen0.item():
                        strengthen_needed_tensor = torch.tensor(True, device=device)
                        action_log_message = f"Strengthening bit 0 (orig_ratio={original_ratio.item():.4f}, alpha_t={alpha_t.item():.4f})"
                    elif condition_strengthen1.item():  # Условие для усиления бита 1
                        strengthen_needed_tensor = torch.tensor(True, device=device)
                        action_log_message = f"Strengthening bit 1 (orig_ratio={original_ratio.item():.4f}, inv_a={inv_a.item():.4f})"
                else:  # modify_needed_tensor is True
                    action_log_message = f"Modifying bit {current_bit_tensor.item()} -> {bit_to_embed_tensor.item()} (tensor)"

                if modify_needed_tensor.item() or strengthen_needed_tensor.item():
                    modified = True
                    energy_sq = s1 ** 2 + s2 ** 2
                    energy = torch.sqrt(energy_sq + eps)  # Добавляем eps для стабильности sqrt, если s1 и s2 оба 0

                    if bit_to_embed_tensor == 0:
                        target_ratio_tensor = alpha_t
                    else:  # bit_to_embed_tensor == 1
                        target_ratio_tensor = inv_a

                    denominator_sq = target_ratio_tensor ** 2 + one_tensor
                    denominator = torch.sqrt(denominator_sq + eps)

                    ns1 = energy * target_ratio_tensor / denominator
                    ns2 = energy / denominator

                    if not (torch.isfinite(ns1) and torch.isfinite(ns2)):
                        logging.warning(
                            f"{prefix} Non-finite ns1 ({ns1.item()}) or ns2 ({ns2.item()}) after calculation. Modification skipped.")
                        modified = False

                logging.info(f"{prefix} ACTION: {action_log_message}.")

                if modified:
                    modifications_count += 1

                    # Реконструкция DCT коэффициентов с новыми сингулярными числами
                    # S_diag ожидает 1D вектор для диагонали, ns1/ns2 уже скалярные тензоры
                    # unsqueeze(0) делает их [1], diag создаст матрицу [1,1]
                    S1_mod_diag = torch.diag(ns1.unsqueeze(0))
                    S2_mod_diag = torch.diag(ns2.unsqueeze(0))

                    d1m_tensor = torch.matmul(U1, torch.matmul(S1_mod_diag, Vh1_t)).squeeze(
                        -1)  # squeeze в конце, т.к. U, S_diag, Vh.T -> (N,1)
                    d2m_tensor = torch.matmul(U2, torch.matmul(S2_mod_diag, Vh2_t)).squeeze(-1)

                    # Обратное DCT
                    v1m_tensor = idct1d_torch(d1m_tensor)
                    v2m_tensor = idct1d_torch(d2m_tensor)

                    if not torch.isfinite(v1m_tensor).all() or not torch.isfinite(v2m_tensor).all():
                        logging.warning(f"{prefix} Non-finite v1m/v2m after IDCT. Skipping update.");
                        continue
                    if v1m_tensor.shape != v1_tensor.shape or v2m_tensor.shape != v2_tensor.shape:  # Проверка формы
                        logging.warning(f"{prefix} Shape mismatch for v1m/v2m after IDCT. Skipping update.");
                        continue

                    # Вычисляем дельту (разницу)
                    delta1_pt = v1m_tensor - v1_tensor
                    delta2_pt = v2m_tensor - v2_tensor

                    # Применяем перцептуальную маску к дельте
                    mf1 = torch.ones_like(delta1_pt, device=device)
                    mf2 = torch.ones_like(delta2_pt, device=device)

                    if use_perceptual_masking and perceptual_mask_tensor is not None:
                        try:
                            # Убеждаемся, что индексы не выходят за пределы маски
                            # perceptual_mask_tensor должен иметь те же размеры, что и Yl_t
                            # rowsX_actual и colsX_actual уже должны быть корректными по длине
                            if perceptual_mask_tensor.shape[0] > rows1_actual.max().item() and \
                                    perceptual_mask_tensor.shape[1] > cols1_actual.max().item():
                                mv1 = perceptual_mask_tensor[rows1_actual, cols1_actual]
                            else:
                                logging.warning(
                                    f"{prefix} Mask shape mismatch or index out of bounds for frame 1 (mask: {perceptual_mask_tensor.shape}, max_row: {rows1_actual.max().item()}, max_col: {cols1_actual.max().item()}). Using ones for mf1.")
                                mv1 = torch.ones_like(rows1_actual, dtype=delta1_pt.dtype, device=device)

                            if perceptual_mask_tensor.shape[0] > rows2_actual.max().item() and \
                                    perceptual_mask_tensor.shape[1] > cols2_actual.max().item():
                                mv2 = perceptual_mask_tensor[rows2_actual, cols2_actual]
                            else:
                                logging.warning(
                                    f"{prefix} Mask shape mismatch or index out of bounds for frame 2. Using ones for mf2.")
                                mv2 = torch.ones_like(rows2_actual, dtype=delta2_pt.dtype, device=device)

                            lambda_t = torch.tensor(LAMBDA_PARAM, device=device, dtype=mf1.dtype)
                            one_minus_lambda_t = torch.tensor(1.0, device=device, dtype=mf1.dtype) - lambda_t
                            mf1 = lambda_t + one_minus_lambda_t * mv1  # Не mf1.mul_, а присваивание нового значения
                            mf2 = lambda_t + one_minus_lambda_t * mv2
                        except IndexError as mask_idx_err:
                            logging.warning(f"{prefix} Mask apply IndexError: {mask_idx_err}. Using unmasked delta.")
                        except Exception as mask_err:
                            logging.warning(f"{prefix} Mask apply general error: {mask_err}. Using unmasked delta.")

                    # Применение модифицированной дельты к Yl_t_mod и Yl_t1_mod
                    # Используем += как в СТАБИЛЬНОЙ версии
                    Yl_t_mod[rows1_actual, cols1_actual] += delta1_pt * mf1
                    Yl_t1_mod[rows2_actual, cols2_actual] += delta2_pt * mf2

            except Exception as e_ring_loop:
                logging.error(f"{prefix} Error in ring loop (GPU Alpha): {e_ring_loop}", exc_info=True)
                continue  # Переход к следующему кольцу/биту

        logging.debug(f"{prefix_base} --- End Ring Loop (GPU Alpha, {modifications_count} modifications) ---")

        # 5. Обратное DTCWT
        # Убедимся, что тензоры имеют правильную размерность (Batch, Channel, H, W) для IDTCWT
        if Yl_t_mod.dim() == 2:
            Yl_t_mod_final = Yl_t_mod.unsqueeze(0).unsqueeze(0)
        elif Yl_t_mod.dim() == 4 and Yl_t_mod.shape[0] == 1 and Yl_t_mod.shape[1] == 1:
            Yl_t_mod_final = Yl_t_mod
        else:
            logging.error(f"Unexpected Yl_t_mod shape: {Yl_t_mod.shape}"); return None, None

        if Yl_t1_mod.dim() == 2:
            Yl_t1_mod_final = Yl_t1_mod.unsqueeze(0).unsqueeze(0)
        elif Yl_t1_mod.dim() == 4 and Yl_t1_mod.shape[0] == 1 and Yl_t1_mod.shape[1] == 1:
            Yl_t1_mod_final = Yl_t1_mod
        else:
            logging.error(f"Unexpected Yl_t1_mod shape: {Yl_t1_mod.shape}"); return None, None

        c1m_np = dtcwt_pytorch_inverse(Yl_t_mod_final, Yh_t, dtcwt_inv, device, target_shape_hw, frame_number)
        c2m_np = dtcwt_pytorch_inverse(Yl_t1_mod_final, Yh_t1, dtcwt_inv, device, target_shape_hw, frame_number + 1)

        if c1m_np is None or c2m_np is None:
            logging.error(f"{prefix_base} DTCWT inverse failed.")
            return None, None

        # 6. Постобработка и сборка кадра
        # Денормализация ([0,1] -> [0,255]), клиппинг и конвертация в uint8
        c1s_np = np.clip(c1m_np * 255.0, 0, 255).astype(np.uint8)
        c2s_np = np.clip(c2m_np * 255.0, 0, 255).astype(np.uint8)

        # Ресайз, если форма не совпадает (маловероятно, если target_shape_hw правильно передана в IDTCWT)
        if c1s_np.shape != target_shape_hw:
            c1s_np = cv2.resize(c1s_np, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_LINEAR)
        if c2s_np.shape != target_shape_hw:
            c2s_np = cv2.resize(c2s_np, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_LINEAR)

        # Копируем исходные YCrCb кадры и заменяем только обработанный компонент
        f1_ycrcb_out_np = f1_ycrcb_np.copy()
        f2_ycrcb_out_np = f2_ycrcb_np.copy()

        f1_ycrcb_out_np[:, :, embed_component] = c1s_np
        f2_ycrcb_out_np[:, :, embed_component] = c2s_np

        # Конвертация обратно в BGR
        f1m = cv2.cvtColor(f1_ycrcb_out_np, cv2.COLOR_YCrCb2BGR)
        f2m = cv2.cvtColor(f2_ycrcb_out_np, cv2.COLOR_YCrCb2BGR)

        logging.debug(f"{prefix_base} Embed Pair (GPU Alpha) Finished Successfully.")
        return f1m, f2m

    except Exception as e_embed_pair:
        logging.error(f"{prefix_base} Critical error in embed_frame_pair (GPU Alpha): {e_embed_pair}", exc_info=True)
        # Очистка CUDA памяти при ошибке, если используется GPU
        if device.type == 'cuda':
            with torch.no_grad():  # Убеждаемся, что нет активных градиентов
                torch.cuda.empty_cache()
        return None, None


# @profile
def _embed_single_pair_task(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """
    Обрабатывает одну пару кадров: выбирает кольца (с GPU энтропией), вызывает embed_frame_pair (PyTorch).
    """
    pair_idx = args.get('pair_idx', -1);
    f1_bgr = args.get('frame1');
    f2_bgr = args.get('frame2')
    bits_for_this_pair = args.get('bits', [])
    nr = args.get('n_rings', N_RINGS);
    nrtu = args.get('num_rings_to_use', NUM_RINGS_TO_USE)
    cps = args.get('candidate_pool_size', CANDIDATE_POOL_SIZE);
    ec = args.get('embed_component', EMBED_COMPONENT)
    upm = args.get('use_perceptual_masking', USE_PERCEPTUAL_MASKING)
    device = args.get('device');
    dtcwt_fwd = args.get('dtcwt_fwd');
    dtcwt_inv = args.get('dtcwt_inv')
    fn = 2 * pair_idx;
    selected_rings = []

    if pair_idx == -1 or f1_bgr is None or f2_bgr is None or not bits_for_this_pair \
            or device is None or dtcwt_fwd is None or dtcwt_inv is None:
        logging.error(f"Missing args or data for _embed_single_pair_task (P:{pair_idx})")
        return fn, None, None, []
    if not PYTORCH_WAVELETS_AVAILABLE:
        logging.error(f"PyTorch Wavelets not available in _embed_single_pair_task (P:{pair_idx})")
        return fn, None, None, []

    try:
        candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
        if not candidate_rings:
            logging.warning(f"[P:{pair_idx}] No candidate rings generated.")
            return fn, f1_bgr, f2_bgr, []

        actual_nrtu = min(nrtu, len(candidate_rings))
        if len(candidate_rings) < nrtu:
            logging.warning(
                f"[P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{nrtu}. Using all available {len(candidate_rings)}.")

        f1_ycrcb_np = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
        comp1_tensor = torch.from_numpy(f1_ycrcb_np[:, :, ec].copy()).to(device=device, dtype=torch.float32) / 255.0

        Yl_t_select, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
        if Yl_t_select is None: raise RuntimeError(f"DTCWT FWD failed for ring selection P:{pair_idx}")
        if Yl_t_select.dim() > 2: Yl_t_select = Yl_t_select.squeeze(0).squeeze(0)

        coords_list_gpu = ring_division(Yl_t_select, nr, fn)
        if coords_list_gpu is None or len(coords_list_gpu) != nr:
            raise RuntimeError(f"GPU Ring division failed or returned incorrect number of rings P:{pair_idx}")

        entropies = []
        min_pixels_for_entropy = 10

        for r_idx_candidate in candidate_rings:
            entropy_val_tensor = torch.tensor(-float('inf'), device=device, dtype=Yl_t_select.dtype)
            if 0 <= r_idx_candidate < len(coords_list_gpu) and \
                    coords_list_gpu[r_idx_candidate] is not None and \
                    coords_list_gpu[r_idx_candidate].shape[0] >= min_pixels_for_entropy:

                coords_tensor_current_ring = coords_list_gpu[r_idx_candidate]
                try:
                    rows_current_ring, cols_current_ring = coords_tensor_current_ring[:, 0], coords_tensor_current_ring[
                                                                                             :, 1]
                    rv_tensor_current_ring = Yl_t_select[rows_current_ring, cols_current_ring]
                    shannon_entropy_tensor, _ = calculate_entropies_torch(rv_tensor_current_ring.float(), fn,
                                                                          r_idx_candidate)  # Убедимся, что float

                    if torch.isfinite(shannon_entropy_tensor):
                        entropy_val_tensor = shannon_entropy_tensor
                except Exception as e_entropy:
                    logging.warning(f"[P:{pair_idx},R_cand:{r_idx_candidate}] PyTorch Entropy calc error: {e_entropy}")
            entropies.append((entropy_val_tensor, r_idx_candidate))

        entropies.sort(key=lambda x: x[0].item(), reverse=True)
        selected_rings = [idx for e_tensor, idx in entropies if e_tensor.item() > -float('inf')][:actual_nrtu]

        if len(selected_rings) < actual_nrtu:
            logging.warning(
                f"[P:{pair_idx}] Fallback ring selection after entropy filtering ({len(selected_rings)}<{actual_nrtu}).")
            for r_cand in candidate_rings:
                if len(selected_rings) >= actual_nrtu: break
                if r_cand not in selected_rings: selected_rings.append(r_cand)

        if not selected_rings and bits_for_this_pair:
            logging.error(f"[P:{pair_idx}] No rings selected, but bits need to be embedded. Returning original frames.")
            return fn, f1_bgr, f2_bgr, []

        logging.info(f"[P:{pair_idx}] Selected rings by GPU entropy: {selected_rings}")

        bits_to_embed_now = bits_for_this_pair[:len(selected_rings)]
        if not selected_rings:
            logging.debug(f"P:{pair_idx} No rings selected, returning original frames.")
            return fn, f1_bgr, f2_bgr, []
        if not bits_to_embed_now and selected_rings:  # Это условие, по идее, не должно срабатывать если selected_rings не пуст
            logging.warning(f"P:{pair_idx} Rings selected but no bits to embed. Returning original frames.")
            return fn, f1_bgr, f2_bgr, selected_rings

        mod_f1, mod_f2 = embed_frame_pair(
            f1_bgr, f2_bgr, bits_to_embed_now, selected_rings, nr, fn, upm, ec,
            device, dtcwt_fwd, dtcwt_inv
        )

        return fn, mod_f1, mod_f2, selected_rings

    except Exception as e:
        logging.error(f"Error in _embed_single_pair_task (GPU entropy) P:{pair_idx}: {e}", exc_info=True)
        return fn, None, None, []


def _embed_batch_worker(batch_args_list: List[Dict]) -> List[
    Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]]:
    batch_results = []
    for args in batch_args_list:
        result = _embed_single_pair_task(args)
        batch_results.append(result)
    return batch_results


@profile
def embed_watermark_in_video(
        frames_to_process: List[np.ndarray],
        payload_id_bytes: bytes,
        n_rings: int = N_RINGS,
        num_rings_to_use: int = NUM_RINGS_TO_USE,
        bits_per_pair: int = BITS_PER_PAIR,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        # --- Параметры гибридного режима ---
        use_hybrid_ecc: bool = True,
        max_total_packets: int = 15,
        use_ecc_for_first: bool = USE_ECC,
        bch_code: Optional[BCH_TYPE] = BCH_CODE_OBJECT,
        # --- Параметры PyTorch ---
        device: torch.device = torch.device("cpu"),
        dtcwt_fwd: Optional[DTCWTForward] = None,
        dtcwt_inv: Optional[DTCWTInverse] = None,

        max_workers: Optional[int] = MAX_WORKERS,
        use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING,
        embed_component: int = EMBED_COMPONENT

    ) -> Optional[List[np.ndarray]]:
    """
    Основная функция встраивания ЦВЗ в предоставленный список видеокадров ("голова").

    Формирует полную битовую последовательность ЦВЗ (с учетом ECC и/или
    многократного встраивания Raw-пакетов), а затем параллельно обрабатывает
    пары кадров для встраивания этих бит. Использует PyTorch для вычислительно
    интенсивных операций (DTCWT, DCT, SVD).

    Args:
        frames_to_process: Список NumPy массивов (кадры в BGR), представляющих "голову" видео.
        payload_id_bytes: Полезная нагрузка (например, ID) для встраивания, в виде байт.
        n_rings: Общее количество концентрических колец для разделения LL-поддиапазона.
        num_rings_to_use: Количество колец, выбираемых для встраивания `bits_per_pair` бит.
        bits_per_pair: Количество бит, встраиваемых в одну пару кадров.
        candidate_pool_size: Размер пула колец-кандидатов, из которых выбираются
                             наилучшие по энтропии.
        use_hybrid_ecc: Если True, первый пакет полезной нагрузки кодируется с ECC
                        (если use_ecc_for_first=True и ECC доступен), а последующие
                        копии встраиваются как Raw-данные.
        max_total_packets: Общее желаемое количество пакетов ЦВЗ (первый + Raw-копии).
        use_ecc_for_first: Если True и use_hybrid_ecc=True, попытаться применить ECC
                           к первому пакету.
        bch_code: Объект galois.BCH для ECC кодирования.
        device: Устройство PyTorch (`torch.device`) для вычислений (CPU или CUDA).
        dtcwt_fwd: Предварительно инициализированный объект DTCWTForward (pytorch_wavelets).
        dtcwt_inv: Предварительно инициализированный объект DTCWTInverse (pytorch_wavelets).
        max_workers: Максимальное количество потоков для ThreadPoolExecutor.
        use_perceptual_masking: Использовать ли перцептуальную маску для модуляции
                                силы встраивания.
        embed_component: Индекс цветового компонента (0=Y, 1=Cr, 2=Cb в YCrCb),
                         в который производится встраивание.

    Returns:
        Optional[List[np.ndarray]]: Список обработанных NumPy кадров (BGR) "головы"
                                     со встроенным ЦВЗ.
                                     Возвращает None в случае критической ошибки.
    """
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        logging.critical("PyTorch Wavelets или Torch DCT недоступны!")
        return None
    if dtcwt_fwd is None or dtcwt_inv is None:
        logging.critical("Экземпляры DTCWTForward/DTCWTInverse не переданы!")
        return None

    num_frames_head = len(frames_to_process)
    total_pairs_head = num_frames_head // 2
    payload_len_bytes = len(payload_id_bytes)
    payload_len_bits = payload_len_bytes * 8

    if payload_len_bits == 0:
        logging.error("Payload ID пустой! Встраивание отменено.")
        return None
    if total_pairs_head == 0:
        logging.warning("Нет пар кадров для обработки в переданном списке.")
        return None
    if max_total_packets <= 0:
         logging.warning(f"max_total_packets ({max_total_packets}) должен быть > 0. Установлено в 1.")
         max_total_packets = 1

    logging.info(f"--- Embed Head Start (Processing {num_frames_head} frames / {total_pairs_head} pairs) ---")
    logging.info(f"Using max_workers: {max_workers}")

    # --- Формирование последовательности бит для встраивания ---
    bits_to_embed_list = []
    raw_payload_bits: Optional[np.ndarray] = None
    try:
        raw_payload_bits = np.unpackbits(np.frombuffer(payload_id_bytes, dtype=np.uint8))
        if raw_payload_bits.size != payload_len_bits:
             raise ValueError(f"Ошибка unpackbits: {payload_len_bits} vs {raw_payload_bits.size}")
    except Exception as e:
        logging.error(f"Ошибка подготовки raw_payload_bits: {e}", exc_info=True)
        return None
    if raw_payload_bits is None: logging.error("Не создан raw_payload_bits."); return None

    first_packet_bits: Optional[np.ndarray] = None
    packet1_type_str = "N/A"; packet1_len = 0; num_raw_packets_added = 0
    can_use_ecc = use_ecc_for_first and GALOIS_AVAILABLE and bch_code is not None and payload_len_bits <= bch_code.k

    if use_hybrid_ecc and can_use_ecc:
        first_packet_bits = add_ecc(raw_payload_bits, bch_code)
        if first_packet_bits is not None:
            bits_to_embed_list.extend(first_packet_bits.tolist())
            packet1_len = len(first_packet_bits); packet1_type_str = f"ECC(n={packet1_len}, t={bch_code.t})"
            logging.info(f"Гибридный режим: Первый пакет создан как {packet1_type_str}.")
        else: logging.error("Ошибка создания ECC пакета!"); return None
    else:
        first_packet_bits = raw_payload_bits; bits_to_embed_list.extend(first_packet_bits.tolist())
        packet1_len = len(first_packet_bits); packet1_type_str = f"Raw({packet1_len})"
        if not use_hybrid_ecc: logging.info(f"Режим НЕ гибридный: Первый пакет - {packet1_type_str}.")
        elif not can_use_ecc: logging.info(f"Гибридный режим: ECC невозможен/выключен. Первый пакет - {packet1_type_str}.")
        use_hybrid_ecc = False

    if use_hybrid_ecc:
        num_raw_repeats_to_add = max(0, max_total_packets - 1)
        for _ in range(num_raw_repeats_to_add):
            bits_to_embed_list.extend(raw_payload_bits.tolist())
            num_raw_packets_added += 1
        if num_raw_packets_added > 0: logging.info(f"Гибридный режим: Добавлено {num_raw_packets_added} Raw пакетов.")

    total_packets_actual = 1 + num_raw_packets_added
    total_bits_to_embed = len(bits_to_embed_list)
    if total_bits_to_embed == 0: logging.error("Нет бит для встраивания."); return None

    if bits_per_pair <= 0: logging.error(f"Invalid bits_per_pair: {bits_per_pair}"); return None
    pairs_needed = ceil(total_bits_to_embed / bits_per_pair)
    # Обрабатываем не больше пар, чем есть в переданных кадрах "головы"
    pairs_to_process = min(total_pairs_head, pairs_needed)

    bits_flat_final = np.array(bits_to_embed_list[:pairs_to_process * bits_per_pair], dtype=np.uint8)
    actual_bits_embedded = len(bits_flat_final)

    logging.info(f"Head processing details:")
    logging.info(f"  Target packets: {total_packets_actual} ({packet1_type_str} + {num_raw_packets_added} Raw)")
    logging.info(f"  Total bits prepared: {total_bits_to_embed}")
    logging.info(f"  Available pairs in head: {total_pairs_head}, Pairs needed: {pairs_needed}, Pairs to process: {pairs_to_process}")
    logging.info(f"  Actual bits to embed in head: {actual_bits_embedded}")
    if pairs_to_process < pairs_needed:
         logging.warning(f"Not enough frames in the provided head ({num_frames_head}) to embed all prepared bits!")

    start_time_embed_loop = time.time()
    watermarked_frames = [frame.copy() for frame in frames_to_process]
    rings_log: Dict[int, List[int]] = {}
    pc, ec, uc = 0, 0, 0
    skipped_pairs = 0
    all_pairs_args = []

    for pair_idx in range(pairs_to_process):
        i1 = 2 * pair_idx
        i2 = i1 + 1
        if i2 >= len(frames_to_process) or frames_to_process[i1] is None or frames_to_process[i2] is None:
            skipped_pairs += 1
            logging.warning(f"Skipping pair {pair_idx}: invalid frames/indices within the head list.")
            continue

        sbi = pair_idx * bits_per_pair
        ebi = sbi + bits_per_pair
        if sbi >= len(bits_flat_final): break
        if ebi > len(bits_flat_final): ebi = len(bits_flat_final)
        cb = bits_flat_final[sbi:ebi].tolist()
        if len(cb) == 0: continue

        args = {'pair_idx': pair_idx,
                'frame1': frames_to_process[i1],
                'frame2': frames_to_process[i2],
                'bits': cb,
                'n_rings': n_rings, 'num_rings_to_use': num_rings_to_use,
                'candidate_pool_size': candidate_pool_size,
                'frame_number': i1,
                'use_perceptual_masking': use_perceptual_masking,
                'embed_component': embed_component,
                'device': device, 'dtcwt_fwd': dtcwt_fwd, 'dtcwt_inv': dtcwt_inv}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if skipped_pairs > 0: logging.warning(f"Skipped {skipped_pairs} pairs during task prep.")
    if num_valid_tasks == 0: logging.error("No valid tasks to process."); return None

    # --- Запуск ThreadPoolExecutor с ОГРАНИЧЕННЫМ числом воркеров ---
    num_workers_to_use = max_workers if max_workers is not None and max_workers > 0 else 1 # Минимум 1
    batch_size = max(1, ceil(num_valid_tasks / num_workers_to_use));
    num_batches = ceil(num_valid_tasks / batch_size)
    batched_args_list = [all_pairs_args[i:i + batch_size] for i in range(0, num_valid_tasks, batch_size) if all_pairs_args[i:i+batch_size]]
    actual_num_batches = len(batched_args_list)

    logging.info(f"Launching {actual_num_batches} batches ({num_valid_tasks} pairs) in ThreadPool (max_workers={num_workers_to_use}, batch≈{batch_size})...")

    try:
        with ThreadPoolExecutor(max_workers=num_workers_to_use) as executor: # Используем num_workers_to_use
            future_to_batch_idx = {executor.submit(_embed_batch_worker, batch): i for i, batch in enumerate(batched_args_list)}
            for future in concurrent.futures.as_completed(future_to_batch_idx):
                batch_idx = future_to_batch_idx[future]
                original_batch = batched_args_list[batch_idx]
                try:
                    batch_results = future.result()
                    if not isinstance(batch_results, list) or len(batch_results) != len(original_batch):
                        logging.error(f"Batch {batch_idx} result size mismatch!")
                        ec += len(original_batch); continue

                    for i, single_res in enumerate(batch_results):
                        original_args = original_batch[i]
                        pair_idx = original_args.get('pair_idx', -1)
                        if pair_idx == -1: logging.error(f"No pair_idx in result?"); ec += 1; continue

                        if isinstance(single_res, tuple) and len(single_res) == 4:
                            fn_res, mod_f1, mod_f2, sel_rings = single_res
                            i1 = 2 * pair_idx; i2 = i1 + 1
                            if isinstance(sel_rings, list): rings_log[pair_idx] = sel_rings
                            if isinstance(mod_f1, np.ndarray) and isinstance(mod_f2, np.ndarray):
                                # Обновляем кадры ВНУТРИ списка watermarked_frames (который является копией "головы")
                                if i1 < len(watermarked_frames): watermarked_frames[i1] = mod_f1; uc += 1
                                else: logging.error(f"Index {i1} out of bounds for watermarked_frames (len={len(watermarked_frames)})")
                                if i2 < len(watermarked_frames): watermarked_frames[i2] = mod_f2; uc += 1
                                else: logging.error(f"Index {i2} out of bounds for watermarked_frames (len={len(watermarked_frames)})")
                                pc += 1
                            else: logging.warning(f"Embedding failed for pair {pair_idx}."); ec += 1
                        else: logging.warning(f"Incorrect result structure for pair {pair_idx}."); ec += 1
                except Exception as e:
                    failed_pairs_count = len(original_batch)
                    logging.error(f"Batch {batch_idx} execution failed: {e}", exc_info=True)
                    ec += failed_pairs_count
    except Exception as e:
        logging.critical(f"ThreadPoolExecutor critical error: {e}", exc_info=True)
        return None

    # --- Завершение и запись логов колец ---
    processing_time = time.time() - start_time_embed_loop
    logging.info(f"Processing {pairs_to_process} pairs finished in {processing_time:.2f} sec.")
    logging.info(f"Result: Processed OK: {pc}, Errors/Skipped: {ec + skipped_pairs}, Frames Updated: {uc}.")

    # Запись лога колец
    if rings_log:
        try:
            serializable_log = {str(k): v for k, v in rings_log.items()}
            with open(SELECTED_RINGS_FILE, 'w', encoding='utf-8') as f: json.dump(serializable_log, f, indent=4)
            logging.info(f"Selected rings log saved: {SELECTED_RINGS_FILE}")
        except Exception as e: logging.error(f"Failed to save rings log: {e}", exc_info=True)
    else: logging.warning("Rings log is empty.")

    logging.info(f"Function embed_watermark_in_video (head processing) finished.")
    return watermarked_frames

# @profile
def main() -> int:
    """
        Основная управляющая функция для процесса встраивания ЦВЗ.
        Выполняет все шаги: от анализа входного видео и подготовки данных
        до встраивания ЦВЗ в "голову", создания "переходного" сегмента (если нужен)
        и "хвоста", и финальной склейки всех частей с помощью FFmpeg.
        """
    main_start_time = time.time()
    logging.info(f"--- Запуск Основного Процесса Встраивания (Smart Stitch) ---")

    # --- Этап 0: Проверки доступности и инициализация ---
    if not PYAV_AVAILABLE:
        logging.critical("PyAV недоступен! Невозможно продолжить.")
        return 1
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        logging.critical("PyTorch Wavelets или Torch DCT недоступны! Невозможно продолжить.")
        return 1
    if USE_ECC and not GALOIS_AVAILABLE:
        logging.warning("ECC включен, но библиотека galois недоступна или не инициализирована. Первый пакет будет Raw.")

    # Настройка PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device=device)
            logging.info(f"Используется CUDA: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e_cuda:
            logging.error(f"Ошибка CUDA при инициализации: {e_cuda}. Переключение на CPU.")
            device = torch.device("cpu")
    else:
        logging.info("Используется CPU.")

    # Создание экземпляров DTCWT
    dtcwt_fwd: Optional[DTCWTForward] = None
    dtcwt_inv: Optional[DTCWTInverse] = None
    if PYTORCH_WAVELETS_AVAILABLE:
        try:
            dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
            dtcwt_inv = DTCWTInverse(biort='near_sym_a', qshift='qshift_a').to(device)
            logging.info("Экземпляры DTCWTForward и DTCWTInverse успешно созданы и перемещены на устройство.")
        except Exception as e_dtcwt_init:
            logging.critical(f"Не удалось инициализировать DTCWT объекты: {e_dtcwt_init}", exc_info=True)
            return 1
    else:
        logging.critical("pytorch_wavelets недоступен! Невозможно создать DTCWT объекты.")
        return 1

    input_video_path = "large.mp4"
    if not os.path.exists(input_video_path):
        logging.critical(f"Входной файл не найден: {input_video_path}")
        print(f"ОШИБКА: Входной файл не найден: {input_video_path}")
        return 1

    base_output_filename = f"watermarked_ffmpeg_t{BCH_T}"

    logging.info(f"Входное видео: '{input_video_path}'")
    logging.info(f"Базовое имя выходного файла: '{base_output_filename}'")

    # Получение метаданных (PyAV -> OpenCV -> MediaInfo)
    logging.info(f"Чтение метаданных для '{input_video_path}'...")
    input_metadata = get_input_metadata(input_video_path)

    if input_metadata is None:
        logging.warning("get_input_metadata (PyAV/OpenCV) не смогла получить метаданные.")
        input_metadata = {}

    if PYMEDIAINFO_AVAILABLE and MediaInfo.can_parse():
        critical_fields_ok = (
                input_metadata.get('width', 0) > 0 and
                input_metadata.get('height', 0) > 0 and
                input_metadata.get('fps') is not None and float(input_metadata.get('fps', 0)) > 0 and
                input_metadata.get('video_codec') is not None
        )
        if not critical_fields_ok:
            logging.info("Попытка получить/дополнить метаданные с помощью pymediainfo...")
            try:
                media_info_obj = MediaInfo.parse(input_video_path)
                video_track = next((t for t in media_info_obj.tracks if t.track_type == 'Video'), None)
                audio_track = next((t for t in media_info_obj.tracks if t.track_type == 'Audio'), None)

                if video_track:
                    if not input_metadata.get('width') and video_track.width: input_metadata[
                        'width'] = video_track.width; logging.info(f"  MediaInfo: Ширина: {video_track.width}")
                    if not input_metadata.get('height') and video_track.height: input_metadata[
                        'height'] = video_track.height; logging.info(f"  MediaInfo: Высота: {video_track.height}")
                    if not input_metadata.get('fps') and video_track.frame_rate:
                        try:
                            input_metadata['fps'] = float(video_track.frame_rate); logging.info(
                                f"  MediaInfo: FPS: {input_metadata['fps']}")
                        except ValueError:
                            logging.warning(f"  MediaInfo: Не удалось преобразовать FPS '{video_track.frame_rate}'")
                    if not input_metadata.get('video_codec') and video_track.codec_id:
                        codec_map = {'avc1': 'h264', 'h264': 'h264', 'hev1': 'hevc', 'h265': 'hevc', 'mp4v': 'mpeg4',
                                     'vp09': 'vp9'}
                        guessed_codec = video_track.format.lower() if video_track.format else (
                            codec_map.get(video_track.codec_id.lower()) if video_track.codec_id else None)
                        if guessed_codec: input_metadata['video_codec'] = guessed_codec; logging.info(
                            f"  MediaInfo: Видео кодек: {guessed_codec}")
                    if not input_metadata.get('duration') and video_track.duration: input_metadata['duration'] = int(
                        video_track.duration * 1000); logging.info(
                        f"  MediaInfo: Длительность: {input_metadata['duration']} us")
                    if not input_metadata.get('total_frames_estimated', 0) > 0 and video_track.frame_count:
                        input_metadata['total_frames_estimated'] = int(video_track.frame_count); logging.info(
                            f"  MediaInfo: Кадры: {video_track.frame_count}")

                if audio_track:
                    if not input_metadata.get('has_audio'): input_metadata['has_audio'] = True
                    if not input_metadata.get('audio_codec') and audio_track.format: input_metadata[
                        'audio_codec'] = audio_track.format.lower(); logging.info(
                        f"  MediaInfo: Аудио кодек: {input_metadata['audio_codec']}")
                    if not input_metadata.get('audio_rate') and audio_track.sampling_rate: input_metadata[
                        'audio_rate'] = audio_track.sampling_rate; logging.info(
                        f"  MediaInfo: Частота дискр. аудио: {audio_track.sampling_rate}")
                    if not input_metadata.get('audio_layout') and audio_track.channel_s:
                        if audio_track.channel_s == 1:
                            input_metadata['audio_layout'] = 'mono'
                        elif audio_track.channel_s == 2:
                            input_metadata['audio_layout'] = 'stereo'
                        if input_metadata.get('audio_layout'): logging.info(
                            f"  MediaInfo: Раскладка аудио: {input_metadata['audio_layout']}")
                    if not input_metadata.get('audio_bitrate') and audio_track.bit_rate: input_metadata[
                        'audio_bitrate'] = audio_track.bit_rate; logging.info(
                        f"  MediaInfo: Аудио битрейт: {audio_track.bit_rate}")
                elif input_metadata.get('has_audio'):
                    logging.warning("  MediaInfo не нашел аудиопоток, хотя другие методы его обнаружили.")

            except Exception as e_mediainfo_parse:
                logging.error(f"Ошибка при получении метаданных через pymediainfo: {e_mediainfo_parse}", exc_info=True)

    if not (input_metadata and
            input_metadata.get('width', 0) > 0 and
            input_metadata.get('height', 0) > 0 and
            input_metadata.get('fps') is not None and float(input_metadata.get('fps', 0)) > 0 and
            input_metadata.get('video_codec') is not None):
        logging.critical("Не удалось получить критически важные метаданные. Прерывание.")
        return 1

    # Окончательная оценка общего числа кадров
    total_original_frames = input_metadata.get('total_frames_estimated', 0)
    if total_original_frames <= 0:
        duration_us = input_metadata.get('duration')
        fps_val = input_metadata.get('fps')
        if duration_us and fps_val and float(fps_val) > 0:
            try:
                total_original_frames = int(round((float(duration_us) / 1_000_000.0) * float(fps_val)))
                input_metadata['total_frames_estimated'] = total_original_frames
                logging.info(f"Общее число кадров оценено из длительности и FPS: {total_original_frames}")
            except Exception as e_calc_final_frames:
                logging.warning(f"Не удалось рассчитать общее число кадров: {e_calc_final_frames}")
                total_original_frames = -1
        else:
            logging.warning("Невозможно оценить общее число кадров.")
            total_original_frames = -1

    original_audio_bitrate = input_metadata.get('audio_bitrate')
    if original_audio_bitrate:
        logging.info(f"Используемый аудио битрейт (из метаданных): {original_audio_bitrate} bps")
    else:
        logging.info("Аудио битрейт не найден в метаданных.")

    payload_id_for_calc_len = os.urandom(PAYLOAD_LEN_BYTES)
    bits_for_calc_len_list: List[int] = []
    try:
        raw_payload_bits_for_calc: np.ndarray = np.unpackbits(np.frombuffer(payload_id_for_calc_len, dtype=np.uint8))
    except Exception as e_unpack_calc:
        logging.critical(f"Ошибка np.unpackbits при расчете длины головы: {e_unpack_calc}", exc_info=True)
        return 1
    can_use_ecc_for_calc = USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is not None and \
                           (len(raw_payload_bits_for_calc) <= BCH_CODE_OBJECT.k)
    if can_use_ecc_for_calc:
        first_packet_calc = add_ecc(raw_payload_bits_for_calc, BCH_CODE_OBJECT)
        if first_packet_calc is not None:
            bits_for_calc_len_list.extend(first_packet_calc.tolist())
        else:
            logging.warning("Ошибка расчета ECC для определения длины 'головы'."); bits_for_calc_len_list.extend(
                raw_payload_bits_for_calc.tolist())
        num_raw_to_add = max(0, MAX_TOTAL_PACKETS - 1) if MAX_TOTAL_PACKETS > 0 else 0
        for _ in range(num_raw_to_add): bits_for_calc_len_list.extend(raw_payload_bits_for_calc.tolist())
    else:
        num_packets_to_add = MAX_TOTAL_PACKETS if MAX_TOTAL_PACKETS > 0 else 1
        for _ in range(num_packets_to_add): bits_for_calc_len_list.extend(raw_payload_bits_for_calc.tolist())
    total_bits_to_embed_estimation = len(bits_for_calc_len_list)
    if BITS_PER_PAIR <= 0: logging.critical(f"BITS_PER_PAIR ({BITS_PER_PAIR}) <= 0."); return 1
    pairs_needed = math.ceil(
        total_bits_to_embed_estimation / BITS_PER_PAIR) if total_bits_to_embed_estimation > 0 else 0
    frames_to_process = pairs_needed * 2
    if total_original_frames > 0: frames_to_process = min(frames_to_process, total_original_frames)
    if frames_to_process % 2 != 0 and frames_to_process > 0: frames_to_process -= 1

    output_extension, target_video_encoder_lib_for_head, ffmpeg_action_original_audio_tail = \
        check_compatibility_and_choose_output(input_metadata)

    final_output_path = base_output_filename + output_extension
    temp_head_path = base_output_filename + "_head" + output_extension

    logging.info(f"Расчетная 'голова' для ЦВЗ: {frames_to_process} кадров.")
    logging.info(f"Видеокодер для 'головы': {target_video_encoder_lib_for_head}")
    logging.info(f"Финальный файл: '{final_output_path}', Временная голова: '{temp_head_path}'")

    # Обработка случая, когда обрабатывать нечего (frames_to_process == 0)
    if frames_to_process <= 0:
        logging.warning("Нет кадров для обработки ЦВЗ (frames_to_process <= 0).")
        print("ПРЕДУПРЕЖДЕНИЕ: Нет кадров для встраивания ЦВЗ. Копирование оригинального файла...")
        try:
            if not os.path.exists(os.path.dirname(final_output_path)):
                os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            shutil.copy2(input_video_path, final_output_path)
            logging.info(f"Оригинал скопирован в '{final_output_path}'")
            print(f"Оригинал скопирован в: {final_output_path}")
            return 0
        except Exception as e_copy_zero_frames:
            logging.error(f"Ошибка копирования оригинала при frames_to_process=0: {e_copy_zero_frames}", exc_info=True)
            return 1

    # --- Анализ I-кадров оригинала ---
    logging.info("Анализ I-кадров оригинального видео...")
    iframe_start_times_seconds = get_iframe_start_times(input_video_path)
    if not iframe_start_times_seconds:
        logging.warning(f"Не удалось получить времена I-кадров для '{input_video_path}'. "
                        "Переходный сегмент не будет создан. Возможны артефакты на стыке.")
    else:
        logging.info(f"Найдено {len(iframe_start_times_seconds)} I-кадров.")

    # --- Этап 2: Чтение и Обработка "Головы" ---
    logging.info(f"Чтение 'головы' ({frames_to_process} кадров) и всех аудиопакетов...")
    video_stream_idx = input_metadata.get('video_stream_index', 0)
    audio_stream_idx = input_metadata.get('audio_stream_index', -1)
    if not input_metadata.get('has_audio', False): audio_stream_idx = -1

    head_frames_bgr, all_audio_packets = read_processing_head(
        input_video_path, frames_to_process, video_stream_idx, audio_stream_idx
    )

    if head_frames_bgr is None or not head_frames_bgr:
        logging.critical(f"Не удалось прочитать 'голову' из '{input_video_path}'. Прерывание.")
        return 1

    actual_frames_read_for_head = len(head_frames_bgr)
    if actual_frames_read_for_head < frames_to_process:
        logging.warning(
            f"Фактически прочитано кадров ({actual_frames_read_for_head}) меньше, чем запрошено ({frames_to_process}).")
        # Корректируем frames_to_process, если нужно
        frames_to_process = actual_frames_read_for_head
        if frames_to_process % 2 != 0: frames_to_process -= 1
        if frames_to_process <= 0: logging.error("Не осталось четного числа кадров после коррекции."); return 1
        head_frames_bgr = head_frames_bgr[:frames_to_process]

    logging.info(
        f"Прочитано {len(head_frames_bgr)} видеокадров для 'головы'. Собрано {len(all_audio_packets or [])} аудиопакетов.")

    # Генерация ID
    original_id_bytes = os.urandom(PAYLOAD_LEN_BYTES)
    original_id_hex = original_id_bytes.hex()
    logging.info(f"Сгенерирован Payload ID: {original_id_hex}")
    try:
        with open(ORIGINAL_WATERMARK_FILE, "w", encoding='utf-8') as f_id:
            f_id.write(original_id_hex)
        logging.info(f"Оригинальный ID сохранен в: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e_id_save:
        logging.error(f"Не удалось сохранить ID: {e_id_save}", exc_info=True)

    # Встраивание ЦВЗ
    logging.info("Встраивание ЦВЗ в 'голову'...")
    watermarked_head_frames = embed_watermark_in_video(
        frames_to_process=head_frames_bgr, payload_id_bytes=original_id_bytes,
        n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE, bits_per_pair=BITS_PER_PAIR,
        candidate_pool_size=CANDIDATE_POOL_SIZE, use_hybrid_ecc=USE_ECC,
        max_total_packets=MAX_TOTAL_PACKETS, use_ecc_for_first=USE_ECC,
        bch_code=BCH_CODE_OBJECT, device=device, dtcwt_fwd=dtcwt_fwd, dtcwt_inv=dtcwt_inv,
        max_workers=SAFE_MAX_WORKERS, use_perceptual_masking=USE_PERCEPTUAL_MASKING,
        embed_component=EMBED_COMPONENT
    )
    if watermarked_head_frames is None or len(watermarked_head_frames) != len(head_frames_bgr):
        logging.critical("Ошибка при встраивании ЦВЗ в 'голову'. Прерывание.")
        return 1
    logging.info("Встраивание ЦВЗ в 'голову' успешно завершено.")
    del head_frames_bgr;
    gc.collect()

    # --- Этап 3: Запись "Головы" и получение точной длительности + параметров ---
    logging.info("Запись обработанной 'головы' и получение параметров...")
    video_enc_opts_for_head = {
        'preset': 'medium',
        'crf': '20',
        'tune': 'zerolatency'
    }
    logging.info(f"Используются опции видеокодера для головы: {video_enc_opts_for_head}")

    audio_bitrate_for_head_str = str(
        original_audio_bitrate) if original_audio_bitrate and original_audio_bitrate >= 32000 else "128k"
    audio_enc_opts_for_head = {'b:a': audio_bitrate_for_head_str}

    actual_head_duration_sec, head_encoding_params = write_head_only(
        watermarked_head_frames=watermarked_head_frames,
        all_audio_packets=all_audio_packets if all_audio_packets is not None else [],
        input_metadata=input_metadata, temp_head_path=temp_head_path,
        target_video_encoder_lib=target_video_encoder_lib_for_head,
        video_encoder_options=video_enc_opts_for_head,
        audio_encoder_options=audio_enc_opts_for_head
    )
    del watermarked_head_frames;
    del all_audio_packets;
    gc.collect()

    if actual_head_duration_sec is None or head_encoding_params is None or actual_head_duration_sec < 0:
        logging.critical(f"Ошибка при записи 'головы' или получены некорректные данные. Прерывание.")
        if os.path.exists(temp_head_path):
            try:
                os.remove(temp_head_path)
            except OSError:
                pass
        return 1

    logging.info(
        f"Обработанная 'голова' записана: '{temp_head_path}'. Точная видео длительность: {actual_head_duration_sec:.9f} сек.")
    logging.debug(f"Параметры кодирования головы: {head_encoding_params}")

    # --- Этап 4 и 5: "Умная" Склейка через concatenate_smart_stitch ---
    logging.info("Запуск умной склейки (Голова + Переход + Хвост_Копия)...")

    ffmpeg_smart_stitch_success = concatenate_smart_stitch(
        original_input_path=input_video_path,
        temp_head_path=temp_head_path,
        final_output_path=final_output_path,
        head_end_time_sec=actual_head_duration_sec,
        input_metadata=input_metadata,
        iframe_times_sec=iframe_start_times_seconds,
        head_encoding_params=head_encoding_params,
    )

    # --- Этап 6: Очистка и Завершение ---
    if os.path.exists(temp_head_path):
        if ffmpeg_smart_stitch_success:
            try:
                os.remove(temp_head_path)
                logging.info(f"Временный файл 'головы' '{temp_head_path}' успешно удален.")
            except OSError as e_remove_head_final:
                logging.error(
                    f"Не удалось удалить временный файл 'головы' '{temp_head_path}' после успешной склейки: {e_remove_head_final}")
        else:
            logging.warning(f"Временный файл 'головы' '{temp_head_path}' не удален из-за ошибки на этапе склейки.")

    if not ffmpeg_smart_stitch_success:
        logging.critical(f"Ошибка при создании финального файла '{final_output_path}' на этапе умной склейки.")
        if os.path.exists(final_output_path):
            try:
                os.remove(final_output_path)
            except OSError:
                pass
        return 1

    # Финальная проверка файла
    if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 1024:
        logging.info(
            f"Финальный файл успешно создан: '{final_output_path}' (размер: {os.path.getsize(final_output_path)} байт).")
        print(f"\nУспешно! Выходной файл: {final_output_path}")
    else:
        logging.error(f"Финальный файл '{final_output_path}' не найден или слишком мал после умной склейки.")
        return 1

    total_main_time = time.time() - main_start_time
    logging.info(f"--- Общее Время Выполнения Скрипта: {total_main_time:.2f} сек ---")
    print(f"Завершено за {total_main_time:.2f} секунд.")
    if os.path.exists(ORIGINAL_WATERMARK_FILE): print(f"ID для извлечения: {ORIGINAL_WATERMARK_FILE}")
    print(f"Лог файл: {LOG_FILENAME}")
    return 0


if __name__ == "__main__":
    # Настройка логирования
    if not logging.getLogger().handlers:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                            format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
    # Уровень логирования
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    # Проверка ключевых зависимостей
    missing_libs_critical = []
    if not PYAV_AVAILABLE: missing_libs_critical.append("PyAV (av)")
    if not PYTORCH_WAVELETS_AVAILABLE: missing_libs_critical.append("pytorch_wavelets")
    if not TORCH_DCT_AVAILABLE: missing_libs_critical.append("torch-dct")
    try:
        import cv2
    except ImportError:
        missing_libs_critical.append("OpenCV (cv2)")
    try:
        import numpy
    except ImportError:
        missing_libs_critical.append("NumPy")
    try:
        import torch
    except ImportError:
        missing_libs_critical.append("PyTorch")
    if missing_libs_critical:
        error_msg = f"ОШИБКА: Отсутствуют КРИТИЧЕСКИ важные библиотеки: {', '.join(missing_libs_critical)}."
        print(error_msg);
        logging.critical(error_msg);
        sys.exit(1)

    if USE_ECC and not GALOIS_AVAILABLE: print(
        "ПРЕДУПРЕЖДЕНИЕ: Библиотека 'galois' не найдена/не работает. ECC будет недоступен.")
    if not PYMEDIAINFO_AVAILABLE: logging.warning("Библиотека 'pymediainfo' не найдена. MediaInfo fallback недоступен.")

    # Профилирование
    DO_PROFILING = False
    profiler_instance = None
    if DO_PROFILING:
        if 'KERNPROF_VAR' not in os.environ and 'profile' not in globals() and cProfile is not None:
            profiler_instance = cProfile.Profile();
            profiler_instance.enable()
            print("cProfile профилирование включено.")
        elif 'profile' in globals() and callable(globals()['profile']):
            print("line_profiler активен. cProfile не будет запущен.")

    final_exit_code = 1
    try:
        final_exit_code = main()
    except FileNotFoundError as e_fnf_main:
        print(f"\nОШИБКА: Файл не найден в main(): {e_fnf_main}"); logging.critical(
            f"FileNotFoundError в main: {e_fnf_main}", exc_info=True)
    except av.FFmpegError as e_av_main:
        print(f"\nОШИБКА PyAV/FFmpeg в main(): {e_av_main}"); logging.critical(f"av.FFmpegError в main: {e_av_main}",
                                                                               exc_info=True)
    except torch.cuda.OutOfMemoryError as e_oom_main:
        print(f"\nОШИБКА: Недостаточно памяти CUDA: {e_oom_main}"); logging.critical(
            f"torch.cuda.OutOfMemoryError в main: {e_oom_main}", exc_info=True); torch.cuda.empty_cache()
    except ImportError as e_imp_main:
        print(f"\nОШИБКА Импорта в main(): {e_imp_main}"); logging.critical(f"ImportError в main: {e_imp_main}",
                                                                            exc_info=True)
    except Exception as e_global_main:
        print(f"\nКРИТИЧЕСКАЯ НЕОБРАБОТАННАЯ ОШИБКА в main(): {e_global_main}"); logging.critical(
            f"Необработанная ошибка в main: {e_global_main}", exc_info=True)
    finally:
        if DO_PROFILING and profiler_instance is not None:
            profiler_instance.disable()
            print("\n--- Статистика Профилирования (cProfile) ---")
            stats_obj = pstats.Stats(profiler_instance).strip_dirs().sort_stats("cumulative")
            stats_obj.print_stats(30)
            profile_prof_file = f"profile_embed_smart_t{BCH_T}.prof"
            profile_txt_file = f"profile_embed_smart_t{BCH_T}.txt"
            try:
                stats_obj.dump_stats(profile_prof_file)
                with open(profile_txt_file, 'w', encoding='utf-8') as f_pstats:
                    ps = pstats.Stats(profiler_instance, stream=f_pstats).strip_dirs().sort_stats('cumulative')
                    ps.print_stats()
                print(f"Статистика профилирования сохранена: {profile_prof_file}, {profile_txt_file}")
                logging.info(f"Статистика профилирования сохранена: {profile_prof_file}, {profile_txt_file}")
            except Exception as e_pstats_save:
                logging.error(f"Ошибка сохранения статистики профилирования: {e_pstats_save}")

        logging.info(f"Скрипт watermark_embedder.py завершен с кодом выхода {final_exit_code}.")
        print(f"\nСкрипт завершен с кодом выхода {final_exit_code}.")
        sys.exit(final_exit_code)
