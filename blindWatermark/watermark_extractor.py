import cProfile
import concurrent
import pstats
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import random
import logging
import time
import json
import os
import hashlib
# import imagehash

# from line_profiler import profile
# import cProfile
# import pstats
import torch
import torch.nn.functional as F
from galois import BCH
from line_profiler import profile

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False
    class DTCWTForward: pass
    class DTCWTInverse: pass
    logging.error("ОШИБКА: Библиотека pytorch_wavelets не найдена! Установите: pip install pytorch_wavelets")
try:
    import torch_dct as dct_torch
    TORCH_DCT_AVAILABLE = True
except ImportError:
    TORCH_DCT_AVAILABLE = False
    logging.error("ОШИБКА: Библиотека torch-dct не найдена! Установите: pip install torch-dct")


from typing import List, Tuple, Optional, Dict, Any, Iterator
import uuid
from math import ceil
from collections import Counter
import sys

# --- Galois импорты ---
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

# --- Глобальные Параметры ---
LAMBDA_PARAM: float = 0.05
ALPHA_MIN: float = 1.13
ALPHA_MAX: float = 1.28
N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0
EMBED_COMPONENT: int = 2
CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2
NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection'
PAYLOAD_LEN_BYTES: int = 8
USE_ECC: bool = True
BCH_M: int = 8
BCH_T: int = 9
FPS: int = 30
LOG_FILENAME: str = 'watermarking_extract_pytorch.log'
INPUT_EXTENSION: str = '.mp4'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS_EXTRACT: Optional[int] = 14
expect_hybrid_ecc_global = True
MAX_TOTAL_PACKETS_global = 15

BCH_CODE_OBJECT: Optional[BCH_TYPE] = None
GALOIS_AVAILABLE = False


if GALOIS_IMPORTED:
    _test_bch_ok = False; _test_decode_ok = False
    try:
        _test_m = BCH_M; _test_t = BCH_T; _test_n = (1 << _test_m) - 1; _test_d = 2 * _test_t + 1
        logging.info(f"Попытка инициализации Galois BCH с n={_test_n}, d={_test_d} (ожидаемое t={_test_t})")
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)
        if _test_t == 5: expected_k = 215
        elif _test_t == 7: expected_k = 201
        elif _test_t == 9: expected_k = 187
        elif _test_t == 11: expected_k = 173
        elif _test_t == 15: expected_k = 131
        else: logging.error(f"Неизвестное k для t={_test_t}"); expected_k = -1

        if expected_k != -1 and hasattr(_test_bch_galois, 't') and hasattr(_test_bch_galois, 'k') \
           and _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
             logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
             _test_bch_ok = True; BCH_CODE_OBJECT = _test_bch_galois
        else: logging.error(f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t=getattr(_test_bch_galois, 't', 'N/A'), k=getattr(_test_bch_galois, 'k', 'N/A').")

        if _test_bch_ok and BCH_CODE_OBJECT is not None:
            try:
                _n_bits = BCH_CODE_OBJECT.n; _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8); GF2 = galois.GF(2); _dummy_cw_vec = GF2(_dummy_cw_bits)
                _msg, _flips = BCH_CODE_OBJECT.decode(_dummy_cw_vec, errors=True)
                _test_decode_ok = (_flips is not None or _flips == 0); logging.info(f"galois: decode() test {'OK' if _test_decode_ok else 'failed'}.")
            except Exception as decode_err: logging.error(f"galois: decode() test failed: {decode_err}", exc_info=True); _test_decode_ok = False
    except Exception as test_err: logging.error(f"galois: ОШИБКА теста: {test_err}", exc_info=True); BCH_CODE_OBJECT = None; _test_bch_ok = False
    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None
if GALOIS_AVAILABLE: logging.info("galois: Готов к использованию.")
else: logging.warning("galois: НЕ ДОСТУПЕН.")

# --- Настройка логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO, format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование конфигурации ---
logging.info(f"--- Запуск Скрипта Извлечения (PyTorch Wavelets & DCT) ---")
logging.info(f"PyTorch Wavelets Доступно: {PYTORCH_WAVELETS_AVAILABLE}")
logging.info(f"Torch DCT Доступно: {TORCH_DCT_AVAILABLE}")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit")
logging.info(f"ECC Ожидается (для 1-го пак.): {USE_ECC}, Доступен/Работает: {GALOIS_AVAILABLE} (BCH m={BCH_M}, t={BCH_T})")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, N_RINGS_Total={N_RINGS}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS_EXTRACT or 'default'}) с батчингом.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error(f"NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE! Проверьте настройки.")
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE != BITS_PER_PAIR.")

# --- Базовые Функции ---

def dct1d_torch(s_tensor: torch.Tensor) -> torch.Tensor:
    """1D DCT-II используя torch-dct."""
    if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct не доступен")
    return dct_torch.dct(s_tensor, norm='ortho')

def svd_torch_s1(tensor_1d: torch.Tensor) -> Optional[torch.Tensor]:
    """Применяет SVD и возвращает только первое сингулярное число как тензор."""
    try:
        tensor_2d = tensor_1d.unsqueeze(-1)
        s_values = torch.linalg.svdvals(tensor_2d)
        if s_values is None or s_values.numel() == 0: return None
        if not torch.isfinite(s_values[0]): return None
        return s_values[0] # Возвращаем тензор (скаляр)
    except Exception as e:
        logging.error(f"PyTorch SVD error: {e}", exc_info=True)
        return None

def dtcwt_pytorch_forward(yp_tensor: torch.Tensor, xfm: DTCWTForward, device: torch.device, fn: int = -1) -> Tuple[Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Применяет прямое DTCWT PyTorch к одному каналу (2D тензору)."""
    if not PYTORCH_WAVELETS_AVAILABLE: logging.error("PTW unavailable."); return None, None
    if not isinstance(yp_tensor, torch.Tensor) or yp_tensor.ndim != 2: logging.error(f"[F:{fn}] Invalid input tensor."); return None, None
    try:
        yp_tensor = yp_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        xfm = xfm.to(device)
        with torch.no_grad(): Yl, Yh = xfm(yp_tensor)
        if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh: logging.error(f"[F:{fn}] DTCWTForward invalid result."); return None, None
        return Yl, Yh
    except Exception as e: logging.error(f"[F:{fn}] PT DTCWT fwd error: {e}"); return None, None

def ring_division(lp_tensor: torch.Tensor, nr: int = N_RINGS, fn: int = -1) -> List[Optional[torch.Tensor]]:
    """Разбивает 2D PyTorch тензор на N колец (версия из embedder)."""
    if not isinstance(lp_tensor, torch.Tensor) or lp_tensor.ndim != 2: logging.error(f"[F:{fn}] Invalid input for ring_division."); return [None] * nr
    H, W = lp_tensor.shape; device = lp_tensor.device
    if H < 2 or W < 2: logging.warning(f"[F:{fn}] Tensor too small ({H}x{W})"); return [None] * nr
    try:
        rr, cc = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0; distances = torch.sqrt((rr - center_r)**2 + (cc - center_c)**2)
        min_dist, max_dist = torch.tensor(0.0, device=device), torch.max(distances)
        if max_dist < 1e-9: ring_bins = torch.tensor([0.0]*(nr + 1), device=device); ring_bins[1:] = max_dist + 1e-6
        else: ring_bins = torch.linspace(min_dist.item(), (max_dist + 1e-6).item(), nr + 1, device=device)
        ring_indices = torch.zeros_like(distances, dtype=torch.long) - 1
        for i in range(nr):
            mask = (distances >= ring_bins[i]) & (distances < ring_bins[i+1] if i < nr-1 else distances <= ring_bins[i+1])
            ring_indices[mask] = i
        ring_indices[distances < ring_bins[1]] = 0
        rings: List[Optional[torch.Tensor]] = [None] * nr
        for rdx in range(nr):
            coords_tensor = torch.nonzero(ring_indices == rdx, as_tuple=False)
            if coords_tensor.shape[0] > 0: rings[rdx] = coords_tensor.long()
        return rings
    except Exception as e: logging.error(f"Ring division PT error F{fn}: {e}"); return [None] * nr

def calculate_entropies(rv: np.ndarray, fn: int = -1, ri: int = -1) -> Tuple[float, float]:
    """
        Вычисляет шенноновскую энтропию и энтропию столкновений (Реньи 2-го порядка
        в вашей старой реализации) для одномерного NumPy массива значений пикселей кольца.

        Args:
            rv: Одномерный NumPy массив значений пикселей (предположительно нормализованных от 0 до 1).
            fn: Номер кадра (для логирования, опционально).
            ri: Индекс кольца (для логирования, опционально).

        Returns:
            Кортеж (float, float): (шенноновская_энтропия, энтропия_столкновений).
                                   Возвращает (0.0, 0.0), если массив пуст или содержит константу.
        """
    eps=1e-12; shannon_entropy=0.; collision_entropy=0.
    if rv.size > 0:
        rvc = np.clip(rv.copy(), 0.0, 1.0)
        if np.all(rvc == rvc[0]): return 0.0, 0.0
        hist, _ = np.histogram(rvc, bins=256, range=(0., 1.), density=False)
        total_count = rvc.size
        if total_count > 0:
            probabilities = hist / total_count
            p = probabilities[probabilities > eps]
            if p.size > 0:
                shannon_entropy = -np.sum(p * np.log2(p))
                ee = -np.sum(p*np.exp(1.-p)); collision_entropy = ee
    return shannon_entropy, collision_entropy

def get_fixed_pseudo_random_rings(pi: int, nr: int, ps: int) -> List[int]:
    """
        Генерирует детерминированный псевдослучайный набор индексов колец
        на основе индекса пары кадров.

        Args:
            pi: Индекс пары кадров (используется как сид для PRNG).
            nr: Общее количество доступных колец (например, N_RINGS).
            ps: Размер пула кандидатов колец, который нужно сгенерировать
                (например, CANDIDATE_POOL_SIZE).

        Returns:
            List[int]: Список псевдослучайных индексов колец без повторений.
                       Длина списка равна `ps` (или `nr`, если `ps > nr`).
                       Пустой список, если `ps <= 0`.
        """

    if ps <= 0: return []
    if ps > nr: ps = nr
    seed_str = str(pi).encode('utf-8'); hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big'); prng = random.Random(seed_int)
    try: candidate_indices = prng.sample(range(nr), ps)
    except ValueError: candidate_indices = list(range(nr)); prng.shuffle(candidate_indices); candidate_indices = candidate_indices[:ps]
    logging.debug(f"[P:{pi}] Candidates: {candidate_indices}");
    return candidate_indices

def bits_to_bytes(bit_list: List[Optional[int]]) -> Optional[bytes]:
    """
        Конвертирует список бит (0 или 1) в байтовую строку.
        Игнорирует None значения в списке. Дополняет нулями до длины, кратной 8, если необходимо.

        Args:
            bit_list: Список, содержащий целые числа 0, 1 или None.

        Returns:
            Optional[bytes]: Байтовая строка, представляющая входные биты,
                             или None в случае ошибки (например, невалидные символы в битах).
                             Пустая байтовая строка b'', если на входе нет валидных бит.
        """

    valid_bits = [b for b in bit_list if b is not None and b in (0, 1)]
    num_bits = len(valid_bits)
    if num_bits == 0: return b''
    remainder = num_bits % 8
    if remainder != 0: padding_len = 8 - remainder; valid_bits.extend([0] * padding_len); num_bits += padding_len
    byte_array = bytearray()
    for i in range(0, num_bits, 8):
        byte_chunk = valid_bits[i:i+8]
        try:
            if len(byte_chunk) != 8: logging.error(f"Byte chunk error: {byte_chunk}"); return None
            byte_val = int("".join(map(str, byte_chunk)), 2); byte_array.append(byte_val)
        except ValueError: logging.error(f"Invalid symbols in bit chunk: {byte_chunk}"); return None
    return bytes(byte_array)

def decode_ecc(packet_bits_list: List[int], bch_code: Optional[BCH_TYPE], expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    """
        Декодирует пакет бит с использованием предоставленного объекта BCH кода.

        Args:
            packet_bits_list: Список бит (0 или 1), представляющий кодовое слово.
            bch_code: Объект galois.BCH, используемый для декодирования.
            expected_data_len_bytes: Ожидаемая длина полезной нагрузки (данных) в байтах.

        Returns:
            Кортеж (Optional[bytes], int):
                - Декодированная полезная нагрузка в виде байт, или None при ошибке.
                - Количество исправленных ошибок (int), или -1, если декодирование
                  не удалось или ошибки неисправимы.
        """

    if not GALOIS_AVAILABLE or bch_code is None: logging.error("ECC decode called but unavailable."); return None, -1
    n_corrected = -1
    try:
        n=bch_code.n; k=bch_code.k; expected_payload_bits=expected_data_len_bytes*8
        if len(packet_bits_list) != n: logging.error(f"Decode ECC: Bad len {len(packet_bits_list)}!={n}"); return None, -1
        if expected_payload_bits > k: logging.error(f"Decode ECC: Payload {expected_payload_bits}>k {k}"); return None, -1
        packet_bits_np=np.array(packet_bits_list, dtype=np.uint8); GF=bch_code.field; rx_vec=GF(packet_bits_np)
        try: corr_msg_vec, n_corrected = bch_code.decode(rx_vec, errors=True)
        except galois.errors.UncorrectableError: logging.warning("Galois ECC: Uncorrectable."); return None, -1
        corr_k_bits=corr_msg_vec.view(np.ndarray).astype(np.uint8)
        if corr_k_bits.size < expected_payload_bits: logging.error(f"Decode ECC: Decoded len {corr_k_bits.size} < {expected_payload_bits}"); return None, n_corrected
        payload_bits=corr_k_bits[:expected_payload_bits]; payload_bytes = bits_to_bytes(payload_bits.tolist())
        if payload_bytes is None: logging.error("Decode ECC: bits_to_bytes failed."); return None, n_corrected
        logging.info(f"Galois ECC: Decoded, corrected {n_corrected} errors.")
        return payload_bytes, n_corrected
    except Exception as e: logging.error(f"Decode ECC unexpected error: {e}", exc_info=True); return None, -1

@profile
def extract_single_bit(L1_tensor: torch.Tensor, L2_tensor: torch.Tensor, ring_idx: int, n_rings: int, fn: int) -> Optional[int]:
    """
    Извлекает один бит (PyTorch DCT/SVD).
    (BN-PyTorch-Optimized-Corrected - V2 Log)
    """
    pair_index = fn // 2
    prefix = f"[BN P:{pair_index}, R:{ring_idx}]"

    try:
        # --- Шаг 1: Проверка входных данных ---
        if L1_tensor is None or L2_tensor is None or L1_tensor.shape != L2_tensor.shape \
           or not isinstance(L1_tensor, torch.Tensor) or not isinstance(L2_tensor, torch.Tensor) \
           or L1_tensor.ndim != 2 or L2_tensor.ndim != 2 \
           or not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
            logging.warning(f"{prefix} Invalid L1/L2 provided.")
            return None
        device = L1_tensor.device

        # --- Шаг 2: Кольцевое деление ---
        r1c = ring_division(L1_tensor, n_rings, fn)
        r2c = ring_division(L2_tensor, n_rings, fn + 1)
        if r1c is None or r2c is None \
           or not(0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
             logging.warning(f"{prefix} Invalid ring index or ring_division failed.")
             return None
        cd1_tensor = r1c[ring_idx]; cd2_tensor = r2c[ring_idx]
        min_ring_size = 10
        if cd1_tensor is None or cd2_tensor is None or cd1_tensor.shape[0] < min_ring_size or cd2_tensor.shape[0] < min_ring_size:
             logging.debug(f"{prefix} Ring coords None or ring too small (<{min_ring_size}).")
             return None

        # --- Шаг 3: Извлечение значений, DCT, SVD (НА ТЕНЗОРАХ) ---
        try:
            rows1, cols1 = cd1_tensor[:, 0], cd1_tensor[:, 1]
            rows2, cols2 = cd2_tensor[:, 0], cd2_tensor[:, 1]
            rv1_tensor = L1_tensor[rows1, cols1].to(dtype=torch.float32)
            rv2_tensor = L2_tensor[rows2, cols2].to(dtype=torch.float32)
            min_s = min(rv1_tensor.numel(), rv2_tensor.numel())
            if min_s == 0: return None
            if rv1_tensor.numel() != rv2_tensor.numel():
                rv1_tensor = rv1_tensor[:min_s]; rv2_tensor = rv2_tensor[:min_s]

            logging.debug(f"{prefix} rv1 stats: size={rv1_tensor.numel()}, mean={rv1_tensor.mean():.6e}, std={rv1_tensor.std():.6e}")
            logging.debug(f"{prefix} rv2 stats: size={rv2_tensor.numel()}, mean={rv2_tensor.mean():.6e}, std={rv2_tensor.std():.6e}")

            # --- PyTorch DCT ---
            if not TORCH_DCT_AVAILABLE: raise RuntimeError("torch-dct not available")
            d1_tensor = dct1d_torch(rv1_tensor)
            d2_tensor = dct1d_torch(rv2_tensor)
            if not torch.isfinite(d1_tensor).all() or not torch.isfinite(d2_tensor).all(): return None
            # *** ЛОГ: Первые DCT коэффициенты ***
            logging.debug(f"{prefix} DCT done. d1[0]={d1_tensor[0]:.6e}, d2[0]={d2_tensor[0]:.6e}")


            # --- PyTorch SVD ---
            s1_tensor = svd_torch_s1(d1_tensor)
            s2_tensor = svd_torch_s1(d2_tensor)
            if s1_tensor is None or s2_tensor is None: return None
            # *** ЛОГ: Сингулярные числа (тензоры) с высокой точностью ***
            logging.debug(f"{prefix} SVD done. s1_tensor={s1_tensor.item():.8e}, s2_tensor={s2_tensor.item():.8e}")

            s1 = s1_tensor.item()
            s2 = s2_tensor.item()

        except RuntimeError as torch_err:
             logging.error(f"{prefix} PyTorch runtime error during Tensor DCT/SVD: {torch_err}", exc_info=True); return None
        except IndexError:
             logging.warning(f"{prefix} Index error getting ring tensor values."); return None
        except Exception as e:
             logging.error(f"{prefix} Error in Tensor DCT/SVD processing part: {e}", exc_info=True); return None

        # --- Шаг 4: Принятие решения ---
        eps = 1e-12; threshold = 1.0
        if abs(s2) < eps:
             logging.warning(f"{prefix} s2={s2:.2e} is close to zero. Unreliable ratio.")
             return None

        ratio = s1 / s2
        # --- Явное вычисление и логирование сравнения ---
        comparison_result = (ratio >= threshold)
        extracted_bit = 0 if comparison_result else 1

        logging.info(f"{prefix} Decision: s1={s1:.8e}, s2={s2:.8e}, ratio={ratio:.8f}, threshold={threshold}, comparison (ratio >= threshold)={comparison_result}, extracted_bit={extracted_bit}")

        return extracted_bit

    except Exception as e:
        logging.error(f"Unexpected error in extract_single_bit (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
        return None

def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
    """
    Обрабатывает батч задач извлечения: выполняет DTCWT один раз на пару,
    затем вызывает extract_single_bit для выбранных колец.
    """
    batch_results: Dict[int, List[Optional[int]]] = {}
    if not batch_args_list: return {}

    args_example = batch_args_list[0]
    nr = args_example.get('n_rings', N_RINGS)
    nrtu = args_example.get('num_rings_to_use', NUM_RINGS_TO_USE)
    cps = args_example.get('candidate_pool_size', CANDIDATE_POOL_SIZE)
    ec = args_example.get('embed_component', EMBED_COMPONENT)
    device = args_example.get('device')
    dtcwt_fwd = args_example.get('dtcwt_fwd')

    if device is None or dtcwt_fwd is None:
         logging.error("Device или DTCWTForward не переданы в _extract_batch_worker!")
         for args in batch_args_list:
              pair_idx = args.get('pair_idx', -1)
              if pair_idx != -1:
                   batch_results[pair_idx] = [None] * nrtu
         return batch_results

    for args in batch_args_list:
        pair_idx = args.get('pair_idx', -1)
        f1_bgr = args.get('frame1')
        f2_bgr = args.get('frame2')

        current_nrtu = args.get('num_rings_to_use', NUM_RINGS_TO_USE)
        extracted_bits_for_pair: List[Optional[int]] = [None] * current_nrtu

        if pair_idx == -1 or f1_bgr is None or f2_bgr is None:
            logging.error(f"Недостаточно аргументов для обработки pair_idx={pair_idx if pair_idx != -1 else 'unknown'}")
            batch_results[pair_idx] = extracted_bits_for_pair
            continue

        fn = 2 * pair_idx
        L1_tensor: Optional[torch.Tensor] = None
        L2_tensor: Optional[torch.Tensor] = None

        try:
            # --- Шаг 1: Преобразование цвета и DTCWT ---
            if not isinstance(f1_bgr, np.ndarray) or not isinstance(f2_bgr, np.ndarray):
                 logging.warning(f"[BN Worker P:{pair_idx}] Input frames not numpy arrays.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            # Конвертация BGR -> YCrCb -> Компонент -> Тензор [0,1]
            y1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
            y2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2YCrCb)
            c1_np = y1[:, :, ec].copy().astype(np.float32) / 255.0
            c2_np = y2[:, :, ec].copy().astype(np.float32) / 255.0
            comp1_tensor = torch.from_numpy(c1_np).to(device=device)
            comp2_tensor = torch.from_numpy(c2_np).to(device=device)

            # Прямое DTCWT
            Yl_t, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
            Yl_t1, _ = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, fn + 1)

            if Yl_t is None or Yl_t1 is None:
                 logging.warning(f"[BN Worker P:{pair_idx}] DTCWT forward failed.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            if Yl_t.dim() > 2: L1_tensor = Yl_t.squeeze(0).squeeze(0)
            elif Yl_t.dim() == 2: L1_tensor = Yl_t
            else: raise ValueError(f"Invalid Yl_t dim: {Yl_t.dim()}")

            if Yl_t1.dim() > 2: L2_tensor = Yl_t1.squeeze(0).squeeze(0)
            elif Yl_t1.dim() == 2: L2_tensor = Yl_t1
            else: raise ValueError(f"Invalid Yl_t1 dim: {Yl_t1.dim()}")

            if not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
                 raise TypeError(f"L1/L2 not float! L1:{L1_tensor.dtype}, L2:{L2_tensor.dtype}")
            if L1_tensor.shape != L2_tensor.shape:
                 raise ValueError(f"L1/L2 shape mismatch! L1:{L1_tensor.shape}, L2:{L2_tensor.shape}")

            # --- Шаг 2: Выбор колец  ---
            coords = ring_division(L1_tensor, nr, fn)
            if coords is None or len(coords) != nr:
                 logging.warning(f"[BN Worker P:{pair_idx}] Ring division failed.")
                 batch_results[pair_idx] = extracted_bits_for_pair; continue

            candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)
            if len(candidate_rings) < current_nrtu:
                logging.warning(f"[BN Worker P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{current_nrtu}.")
                current_nrtu = len(candidate_rings) # Обновление nrtu до числа кандидатов
            if current_nrtu == 0:
                logging.error(f"[BN Worker P:{pair_idx}] No candidates to select rings from.")
                batch_results[pair_idx] = []; continue

            # Выбор по энтропии
            entropies = []; min_pixels = 10
            L1_numpy_for_entropy = L1_tensor.cpu().numpy()
            for r_idx_cand in candidate_rings:
                entropy_val = -float('inf')
                if 0 <= r_idx_cand < len(coords) and isinstance(coords[r_idx_cand], torch.Tensor) and coords[r_idx_cand].shape[0] >= min_pixels:
                     c_tensor = coords[r_idx_cand]; rows_t, cols_t = c_tensor[:, 0], c_tensor[:, 1]
                     rows_np, cols_np = rows_t.cpu().numpy(), cols_t.cpu().numpy()
                     try:
                         rv_np = L1_numpy_for_entropy[rows_np, cols_np]
                         shannon_entropy, _ = calculate_entropies(rv_np, fn, r_idx_cand)
                         if np.isfinite(shannon_entropy): entropy_val = shannon_entropy
                     except IndexError: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] IndexError entropy")
                     except Exception as e_entr: logging.warning(f"[BN P:{pair_idx} R_cand:{r_idx_cand}] Entropy error: {e_entr}")
                entropies.append((entropy_val, r_idx_cand))

            entropies.sort(key=lambda x: x[0], reverse=True)
            selected_rings = [idx for e, idx in entropies if e > -float('inf')][:current_nrtu]

            # Fallback
            if len(selected_rings) < current_nrtu:
                logging.warning(f"[BN Worker P:{pair_idx}] Fallback ring selection ({len(selected_rings)}<{current_nrtu}).")
                deterministic_fallback = candidate_rings[:current_nrtu]
                needed = current_nrtu - len(selected_rings)
                for ring in deterministic_fallback:
                    if needed == 0: break
                    if ring not in selected_rings:
                        selected_rings.append(ring)
                        needed -= 1
                if len(selected_rings) < current_nrtu:
                     logging.error(f"[BN Worker P:{pair_idx}] Fallback failed, not enough rings ({len(selected_rings)}<{current_nrtu}).")
                     # Установка nrtu равным числу фактически выбранных колец
                     current_nrtu = len(selected_rings)
                     extracted_bits_for_pair = [None] * current_nrtu

            # Логируем финально выбранные кольца
            logging.info(f"[BN Worker P:{pair_idx}] Selected {len(selected_rings)} rings for extraction: {selected_rings}")

            # --- Шаг 3: Извлечение бит из выбранных колец ---
            # размер списка бит соответствует числу выбранных колец
            extracted_bits_for_pair = [None] * len(selected_rings)
            for i, ring_idx_to_extract in enumerate(selected_rings):
                 extracted_bits_for_pair[i] = extract_single_bit(L1_tensor, L2_tensor, ring_idx_to_extract, nr, fn)

            while len(extracted_bits_for_pair) < nrtu:
                 extracted_bits_for_pair.append(None)

            batch_results[pair_idx] = extracted_bits_for_pair[:nrtu]
        except cv2.error as cv_err:
             logging.error(f"OpenCV error P:{pair_idx} in BN worker: {cv_err}", exc_info=True)
             batch_results[pair_idx] = [None] * nrtu
        except RuntimeError as torch_err: # Ловим ошибки PyTorch отдельно
            logging.error(f"PyTorch runtime error P:{pair_idx} in BN worker: {torch_err}", exc_info=True)
            batch_results[pair_idx] = [None] * nrtu
        except Exception as e:
            logging.error(f"Unexpected error processing pair {pair_idx} in BN worker: {e}", exc_info=True)
            batch_results[pair_idx] = [None] * nrtu

    return batch_results


# --- Вспомогательная функция чтения кадров ---
def read_required_frames_opencv(video_path: str, num_frames_to_read: int) -> Optional[List[np.ndarray]]:
    """
    Читает ТОЛЬКО первые num_frames_to_read кадров с помощью OpenCV.
    """
    frames_opencv = []
    cap = None
    logging.info(f"[OpenCV Read Limited] Попытка открыть: '{video_path}' для чтения {num_frames_to_read} кадров")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"[OpenCV Read Limited] Не удалось открыть файл: {video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cv = cap.get(cv2.CAP_PROP_FPS)
        logging.debug(f"[OpenCV Read Limited] Видео: {width}x{height} @ {fps_cv:.2f} FPS")

        for i in range(num_frames_to_read):
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(
                    f"[OpenCV Read Limited] Не удалось прочитать кадр {i} или достигнут конец файла (запрошено {num_frames_to_read}, прочитано {len(frames_opencv)}).")
                break
            frames_opencv.append(frame)
            if (i + 1) % 200 == 0:
                logging.info(f"[OpenCV Read Limited] Прочитано кадров: {i + 1}/{num_frames_to_read}")

        logging.info(f"[OpenCV Read Limited] Чтение завершено. Получено кадров: {len(frames_opencv)}.")

        if len(frames_opencv) == 0 and num_frames_to_read > 0:  # Если ничего не прочитали, но должны были
            logging.error(f"[OpenCV Read Limited] Не удалось прочитать ни одного кадра из '{video_path}'.")
            return None

        return frames_opencv

    except Exception as e:
        logging.error(f"[OpenCV Read Limited] Ошибка при чтении файла '{video_path}': {e}", exc_info=True)
        return None
    finally:
        if cap:
            cap.release()
            logging.debug("[OpenCV Read Limited] VideoCapture освобожден.")


def generate_frame_pairs_opencv(video_path: str,
                                pairs_to_process: int,
                                # Параметры, которые просто передаются дальше в args
                                nr: int, nrtu: int, cps: int, ec: int,
                                device: Optional[torch.device],
                                dtcwt_fwd: Optional[DTCWTForward]
                                ) -> Iterator[Dict[str, Any]]:
    """
    Ленивый генератор, читающий видеофайл с помощью OpenCV (grab/retrieve)
    и выдающий словари с аргументами для обработки пар кадров.

    Args:
        video_path: Путь к видеофайлу.
        pairs_to_process: Максимальное количество пар для генерации.
        nr, nrtu, cps, ec, device, dtcwt_fwd: Параметры для _extract_batch_worker.

    Yields:
        Словарь с аргументами для _extract_batch_worker для каждой пары кадров.
    """
    cap = None
    frames_read_count = 0
    pairs_yielded_count = 0

    logging.info(f"[Генератор OpenCV] Инициализация для '{video_path}', макс. {pairs_to_process} пар.")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видеофайл OpenCV: {video_path}")

        frame_t: Optional[np.ndarray] = None
        frame_t_plus_1: Optional[np.ndarray] = None

        read_success = True

        # Предварительное чтение первого кадра (будущий frame_t для первой пары)
        if read_success:
            read_success = cap.grab()
            if read_success:
                ret_t, frame_t = cap.retrieve()
                read_success = ret_t and isinstance(frame_t, np.ndarray)
                if read_success:
                    frames_read_count += 1
                    logging.debug(f"[Генератор OpenCV] Успешно прочитан первый кадр (индекс 0).")
                else:
                    logging.warning("[Генератор OpenCV] Не удалось получить первый кадр после grab().")
            else:
                logging.warning("[Генератор OpenCV] Не удалось захватить первый кадр (grab failed).")

        # Основной цикл генерации пар
        for pair_idx in range(pairs_to_process):
            if not read_success:  # Если предыдущее чтение было неудачным
                logging.warning(f"[Генератор OpenCV] Предыдущее чтение не удалось, остановка на паре {pair_idx}.")
                break

            grab_success = cap.grab()
            if not grab_success:
                logging.warning(
                    f"[Генератор OpenCV] Не удалось захватить кадр {frames_read_count} для пары {pair_idx} (grab failed). Конец файла?")
                break

            retrieve_success, frame_t_plus_1 = cap.retrieve()
            if not retrieve_success or not isinstance(frame_t_plus_1, np.ndarray):
                logging.warning(
                    f"[Генератор OpenCV] Не удалось получить кадр {frames_read_count} для пары {pair_idx} после grab(). Конец файла или ошибка.")
                break

            frames_read_count += 1

            if frame_t is not None:  # frame_t должен быть не None после первой успешной итерации
                args = {
                    'pair_idx': pair_idx,
                    'frame1': frame_t.copy(),
                    'frame2': frame_t_plus_1.copy(),
                    'n_rings': nr, 'num_rings_to_use': nrtu,
                    'candidate_pool_size': cps, 'embed_component': ec,
                    'device': device, 'dtcwt_fwd': dtcwt_fwd
                }
                yield args
                pairs_yielded_count += 1

                frame_t = frame_t_plus_1
                frame_t_plus_1 = None
            else:
                logging.error(
                    "[Генератор OpenCV] Ошибка логики: frame_t is None внутри основного цикла, пара не может быть сформирована.")
                break

        logging.info(
            f"Генератор OpenCV завершил работу. Выдано пар: {pairs_yielded_count}. Всего прочитано кадров: {frames_read_count}.")

    except IOError as e_io:
        logging.error(f"Ошибка открытия видеофайла в генераторе OpenCV: {e_io}", exc_info=False)
    except Exception as e_gen:
        logging.error(f"Неожиданная ошибка в генераторе OpenCV: {e_gen}", exc_info=True)
    finally:
        if cap and cap.isOpened():
            cap.release()
            logging.debug("Генератор OpenCV: VideoCapture освобожден.")
        final_yield_count = locals().get('pairs_yielded_count', 0)
        logging.debug(f"[Генератор OpenCV] Финальное количество выданных пар: {final_yield_count}")


# --- Основная функция извлечения ---
# @profile
def extract_watermark_from_video(
        frames: List[np.ndarray],
        nr: int = N_RINGS,
        nrtu: int = NUM_RINGS_TO_USE,
        bp: int = BITS_PER_PAIR,
        cps: int = CANDIDATE_POOL_SIZE,
        ec: int = EMBED_COMPONENT,
        expect_hybrid_ecc: bool = True,
        max_expected_packets: int = 15,
        ue: bool = USE_ECC,
        bch_code: Optional[BCH_TYPE] = BCH_CODE_OBJECT,
        device: Optional[torch.device] = None,
        dtcwt_fwd: Optional[DTCWTForward] = None,
        plb: int = PAYLOAD_LEN_BYTES,
        mw: Optional[int] = MAX_WORKERS_EXTRACT
) -> Optional[bytes]:
    """
    Основная функция извлечения ЦВЗ из предоставленного списка кадров.
    Использует ThreadPoolExecutor для параллельной обработки пар.
    """
    if not frames:
        logging.error("Список кадров пуст! Нечего извлекать.")
        return None
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        logging.critical("Отсутствуют PyTorch Wavelets или Torch DCT!")
        return None
    if device is None or dtcwt_fwd is None:
        logging.critical("Device или DTCWTForward не переданы!")
        return None
    if ue and expect_hybrid_ecc and not GALOIS_AVAILABLE:
        logging.error("ECC требуется для гибридного режима, но Galois недоступен!")

    logging.info(f"--- Запуск Извлечения (из предоставленного списка {len(frames)} кадров, Параллельно) ---")
    logging.info(f"Параметры: Hybrid={expect_hybrid_ecc}, MaxPkts={max_expected_packets}, NRTU={nrtu}, BP={bp}")
    start_time = time.time()

    nf = len(frames)
    total_pairs_available = nf // 2
    if total_pairs_available == 0:
        logging.error("В предоставленном списке нет пар кадров для обработки.")
        return None

    # --- Расчет необходимого количества пар и общей длины бит ---
    payload_len_bits = plb * 8
    packet_len_if_ecc = payload_len_bits
    packet_len_if_raw = payload_len_bits
    ecc_possible_for_first = False
    bch_n = 0

    if ue and GALOIS_AVAILABLE and bch_code is not None:
        try:
            n = bch_code.n;
            k = bch_code.k;
            t_bch = bch_code.t
            if payload_len_bits <= k:
                packet_len_if_ecc = n;
                bch_n = n;
                ecc_possible_for_first = True
                logging.info(f"ECC проверка: Возможно для 1-го пакета (n={n}, k={k}, t={t_bch}).")
            else:
                logging.warning(f"ECC проверка: Payload ({payload_len_bits}) > k ({k}).")
        except Exception as e_galois_check:
            logging.error(f"ECC проверка: Ошибка параметров Galois: {e_galois_check}.")
    else:
        logging.info("ECC проверка: Выключен или недоступен.")

    effective_expect_hybrid_ecc = expect_hybrid_ecc
    if effective_expect_hybrid_ecc and not ecc_possible_for_first:
        logging.warning("Гибридный режим запрошен, но ECC для первого пакета невозможен. Переключение на Raw для всех.")
        effective_expect_hybrid_ecc = False

    max_possible_bits = 0
    if effective_expect_hybrid_ecc:
        max_possible_bits = packet_len_if_ecc + max(0, max_expected_packets - 1) * packet_len_if_raw
    else:
        current_packet_len_for_calc = packet_len_if_ecc if ue and ecc_possible_for_first else packet_len_if_raw
        max_possible_bits = max_expected_packets * current_packet_len_for_calc

    if bp <= 0: logging.error("Bits per pair (bp) <= 0!"); return None
    pairs_needed = ceil(max_possible_bits / bp) if max_possible_bits > 0 else 0

    pairs_to_process = min(total_pairs_available, pairs_needed)
    logging.info(f"Цель извлечения: до {max_expected_packets} пакетов (~{max_possible_bits} бит).")
    logging.info(
        f"Пар кадров: Доступно в списке={total_pairs_available}, Нужно={pairs_needed}, Будет обработано={pairs_to_process}")

    if pairs_to_process == 0:
        logging.warning("Нечего обрабатывать (pairs_to_process=0).")
        return None

    # --- Подготовка аргументов для батчей ---
    all_pairs_args = []
    skipped_pairs = 0
    for pair_idx in range(pairs_to_process):
        i1 = 2 * pair_idx
        i2 = i1 + 1
        if i2 >= nf:
            logging.error(
                f"Критическая ошибка: Индекс i2={i2} выходит за пределы списка кадров (длина {nf}) при pair_idx={pair_idx}.")
            break
        frame1 = frames[i1]
        frame2 = frames[i2]
        if frame1 is None or frame2 is None:
            logging.warning(f"Пропуск пары {pair_idx}: один из кадров None в предоставленном списке.")
            skipped_pairs += 1
            continue  # Пропуск эту пары

        args = {'pair_idx': pair_idx,
                'frame1': frame1.copy(),
                'frame2': frame2.copy(),
                'n_rings': nr, 'num_rings_to_use': nrtu,
                'candidate_pool_size': cps, 'embed_component': ec,
                'device': device, 'dtcwt_fwd': dtcwt_fwd}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if num_valid_tasks == 0:
        logging.error("Не создано ни одной валидной задачи для ThreadPoolExecutor.")
        return None
    if skipped_pairs > 0:
        logging.warning(f"Было пропущено {skipped_pairs} пар из-за None кадров.")
    if num_valid_tasks < pairs_to_process:
        logging.warning(
            f"Количество валидных задач ({num_valid_tasks}) меньше, чем изначально планировалось ({pairs_to_process}). Обновляем pairs_to_process.")
        pairs_to_process = num_valid_tasks
        if pairs_to_process == 0:
            logging.error("После пропусков не осталось задач для обработки.")
            return None

    # --- Запуск ThreadPoolExecutor ---
    num_workers = mw if mw is not None and mw > 0 else (os.cpu_count() or 1)
    batch_size = max(1, ceil(pairs_to_process / (num_workers * 4)))  # Адаптивный размер батча
    batched_args_list = [all_pairs_args[i: i + batch_size] for i in range(0, pairs_to_process, batch_size) if
                         all_pairs_args[i:i + batch_size]]
    actual_num_batches = len(batched_args_list)
    logging.info(
        f"Запуск {actual_num_batches} батчей ({pairs_to_process} пар) в ThreadPool (mw={num_workers}, batch_size≈{batch_size})...")

    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures_map: Dict[concurrent.futures.Future, int] = {}
    extracted_bits_map: Dict[int, List[Optional[int]]] = {}

    processed_pairs_from_futures = 0
    errors_in_futures = 0

    try:
        for i, batch in enumerate(batched_args_list):
            first_pair_idx_in_batch = batch[0]['pair_idx'] if batch else (i * batch_size)
            future = executor.submit(_extract_batch_worker, batch)
            futures_map[future] = first_pair_idx_in_batch

        # --- Ожидание результатов ---
        logging.info(f"Все задачи ({len(futures_map)} батчей) отправлены. Ожидание результатов...")
        for future in concurrent.futures.as_completed(futures_map):
            batch_start_pair_index = futures_map.get(future, -1)
            try:
                batch_results_map = future.result()  #Dict[int, List[Optional[int]]]
                if batch_results_map:
                    extracted_bits_map.update(batch_results_map)
                    processed_pairs_from_futures += len(batch_results_map)
                    errors_in_futures += sum(
                        1 for bits_list in batch_results_map.values() if bits_list is None or None in bits_list)
            except Exception as e_future:
                logging.error(
                    f"Ошибка выполнения батча (начинающегося примерно с пары {batch_start_pair_index}): {e_future}",
                    exc_info=True)

    except Exception as e_executor:
        logging.critical(f"Критическая ошибка ThreadPoolExecutor: {e_executor}", exc_info=True)
        if 'executor' in locals() and executor: executor.shutdown(wait=False, cancel_futures=True)
        return None
    finally:
        if 'executor' in locals() and executor:
            executor.shutdown(wait=True)
            logging.debug("ThreadPoolExecutor остановлен.")

    logging.info(f"Обработка задач завершена. Пар с результатом из воркеров: {processed_pairs_from_futures}. "
                 f"Из них с ошибками извлечения (None в битах): {errors_in_futures}.")

    if not extracted_bits_map and pairs_to_process > 0:
        logging.error("Ни одной пары не было успешно обработано (карта результатов пуста).")
        return None

    # --- Сборка бит ---
    extracted_bits_all: List[Optional[int]] = []
    logging.info(f"Сборка бит для {pairs_to_process} запланированных пар...")

    for pair_idx_loop in range(pairs_to_process):
        bits = extracted_bits_map.get(pair_idx_loop)
        if bits and isinstance(bits, list) and len(bits) == bp:
            extracted_bits_all.extend(bits)
        else:
            if pair_idx_loop in extracted_bits_map:
                logging.warning(
                    f"Пара {pair_idx_loop}: Некорректный результат в карте ({len(bits) if isinstance(bits, list) else type(bits)}). Добавляем None * {bp}")
            else:
                logging.debug(
                    f"Пара {pair_idx_loop}: Нет результата в карте (вероятно, ошибка батча или не дошла очередь). Добавляем None * {bp}")
            extracted_bits_all.extend([None] * bp)

    total_bits_collected = len(extracted_bits_all)
    valid_bits = [b for b in extracted_bits_all if b is not None and b in (0, 1)]
    num_valid_bits = len(valid_bits)
    num_error_bits = total_bits_collected - num_valid_bits
    success_rate = (num_valid_bits / total_bits_collected) * 100 if total_bits_collected > 0 else 0
    logging.info(
        f"Сборка бит: Собрано={total_bits_collected}, Валидных={num_valid_bits} ({success_rate:.1f}%), Ошибок/None={num_error_bits}.")
    if not valid_bits: logging.error("Нет валидных бит (0/1) для декодирования."); return None

    # --- Гибридное Декодирование Пакетов ---
    all_payload_attempts_bits: List[Optional[List[int]]] = []
    decoded_success_count = 0;
    decode_failed_count = 0;
    total_corrected_symbols = 0
    num_processed_bits = 0
    print("\n--- Попытки Декодирования Пакетов ---")
    print(f"{'Pkt #':<6} | {'Type':<7} | {'ECC Status':<18} | {'Corrected':<10} | {'Payload (Hex)':<20}");
    print("-" * 68)
    for i in range(max_expected_packets):
        is_first_packet = (i == 0);
        use_ecc_for_this = is_first_packet and expect_hybrid_ecc and ecc_possible_for_first
        current_packet_len = packet_len_if_ecc if use_ecc_for_this else packet_len_if_raw
        packet_type_str = "ECC" if use_ecc_for_this else "Raw"
        start_idx = num_processed_bits;
        end_idx = start_idx + current_packet_len
        if end_idx > num_valid_bits: logging.warning(f"Не хватает бит для пакета {i + 1}."); break
        packet_candidate_bits = valid_bits[start_idx:end_idx];
        num_processed_bits += current_packet_len
        payload_bytes: Optional[bytes] = None;
        payload_bits: Optional[List[int]] = None;
        errors: int = -1
        status_str = f"Failed ({packet_type_str})";
        payload_hex_str = "N/A"
        if use_ecc_for_this:
            if bch_code is not None:
                payload_bytes, errors = decode_ecc(packet_candidate_bits, bch_code, plb)
                if payload_bytes is not None:
                    status_str = f"OK (ECC: {(errors if errors != -1 else 0)} fixed)"; total_corrected_symbols += max(0,
                                                                                                                      errors)
                else:
                    status_str = f"Uncorrectable(ECC)" if errors == -1 else "ECC Decode Err"
            else:
                status_str = "ECC Code Miss"
        else:
            if len(packet_candidate_bits) >= payload_len_bits:
                payload_bytes_raw = bits_to_bytes(packet_candidate_bits[:payload_len_bits])
                if payload_bytes_raw is not None and len(payload_bytes_raw) == plb:
                    payload_bytes = payload_bytes_raw; errors = 0; status_str = "OK (Raw)"
                else:
                    status_str = "Fail (Raw Conv)"
            else:
                status_str = "Fail (Raw Short)"
        if payload_bytes is not None:
            payload_hex_str = payload_bytes.hex()
            try:
                payload_np_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
                if len(payload_np_bits) == payload_len_bits:
                    payload_bits = payload_np_bits.tolist(); decoded_success_count += 1
                else:
                    status_str += "[Len Fail]"; payload_bits = None
            except Exception:
                status_str += "[Unpack Fail]"; payload_bits = None
        if payload_bits is None: decode_failed_count += 1
        all_payload_attempts_bits.append(payload_bits)
        corrected_str = str(errors) if errors >= 0 else "-"
        print(f"{i + 1:<6} | {packet_type_str:<7} | {status_str:<18} | {corrected_str:<10} | {payload_hex_str:<20}")
    print("-" * 68);
    logging.info(
        f"Итоги декодирования: Попыток={len(all_payload_attempts_bits)}, Успешно={decoded_success_count}, Ошибок={decode_failed_count}.")
    if ecc_possible_for_first and expect_hybrid_ecc: logging.info(
        f"Всего исправлено ECC символов: {total_corrected_symbols}.")

    # --- Побитовое Голосование ---
    if not all_payload_attempts_bits: logging.error("Нет пакетов для голосования."); return None
    first_packet_payload = all_payload_attempts_bits[0]
    valid_decoded_payloads = [p for p in all_payload_attempts_bits if p is not None and len(p) == payload_len_bits]
    num_valid_packets_for_vote = len(valid_decoded_payloads)
    if num_valid_packets_for_vote == 0: logging.error(
        f"Нет валидных {payload_len_bits}-бит пакетов для голосования."); return None
    final_payload_bits = []
    logging.info(f"Голосование по {num_valid_packets_for_vote} валидным пакетам...")
    print("\n--- Результаты Голосования по Битам ---")
    print(f"{'Bit Pos':<8} | {'Votes 0':<8} | {'Votes 1':<8} | {'Winner':<8} | {'Tiebreak?':<10}");
    print("-" * 50)
    for j in range(payload_len_bits):
        votes_for_0 = 0;
        votes_for_1 = 0
        for i in range(num_valid_packets_for_vote):
            if j < len(valid_decoded_payloads[i]):
                if valid_decoded_payloads[i][j] == 1:
                    votes_for_1 += 1
                else:
                    votes_for_0 += 1
        winner_bit: Optional[int] = None;
        tiebreak_used = "No"
        valid_votes_count = votes_for_0 + votes_for_1
        if valid_votes_count == 0:
            final_payload_bits = None; logging.error(f"Bit {j}: Нет голосов!"); break
        elif votes_for_1 > votes_for_0:
            winner_bit = 1
        elif votes_for_0 > votes_for_1:
            winner_bit = 0
        else:
            tiebreak_used = "Yes"
            if first_packet_payload is not None and j < len(first_packet_payload):
                winner_bit = first_packet_payload[j]
            else:
                final_payload_bits = None; logging.error(f"Bit {j}: Ничья, не разрешена!"); break
        if winner_bit is None: final_payload_bits = None; logging.error(f"Bit {j}: Winner is None!"); break
        final_payload_bits.append(winner_bit)
        print(f"{j:<8} | {votes_for_0:<8} | {votes_for_1:<8} | {winner_bit:<8} | {tiebreak_used:<10}")
    print("-" * 50)
    if final_payload_bits is None: logging.error("Голосование не удалось."); return None
    logging.info(f"Голосование завершено.")

    # --- Конвертация и возврат результата ---
    final_payload_bytes = bits_to_bytes(final_payload_bits)
    if final_payload_bytes is None: logging.error("Конвертация бит в байты не удалась."); return None
    if len(final_payload_bytes) != plb: logging.error(
        f"Финальная длина ({len(final_payload_bytes)}B) != ожидаемой ({plb}B)."); return None
    logging.info(f"Финальный ID после голосования: {final_payload_bytes.hex()}")
    end_time = time.time();
    logging.info(f"Извлечение завершено. Общее время: {end_time - start_time:.2f} сек.")
    return final_payload_bytes


# --- Функция main ---
def main() -> int:
    """
    Основная функция запуска экстрактора ЦВЗ.
    Читает необходимое количество кадров с помощью OpenCV и передает их
    функции извлечения, которая использует ThreadPoolExecutor.
    """
    main_start_time = time.time()
    logging.info(f"--- Запуск Основного Процесса Извлечения (OpenCV Limited Read + ThreadPool) ---")

    # --- Инициализация PyTorch и Проверки ---
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        print("ERROR: PyTorch libraries required.");
        logging.critical("Критические PyTorch библиотеки не найдены.")
        return 1
    if USE_ECC and not GALOIS_AVAILABLE:
        global_expect_hybrid_ecc = globals().get('expect_hybrid_ecc_global', USE_ECC)
        if global_expect_hybrid_ecc:
            print("\nWARNING: ECC requested for hybrid mode but galois unavailable.")
            logging.warning("Galois недоступен, ECC для гибридного режима не будет работать.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device=device)
            logging.info(f"Используется CUDA: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e_cuda_init:
            logging.error(f"Ошибка CUDA: {e_cuda_init}. Переключение на CPU.")
            device = torch.device("cpu")
    else:
        logging.info("Используется CPU.")

    dtcwt_fwd: Optional[DTCWTForward] = None
    if PYTORCH_WAVELETS_AVAILABLE:
        try:
            dtcwt_fwd = DTCWTForward(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
            logging.info("PyTorch DTCWTForward instance created.")
        except Exception as e_dtcwt:
            logging.critical(f"Failed to init DTCWTForward: {e_dtcwt}", exc_info=True)
            return 1
    else:
        logging.critical("PyTorch Wavelets недоступен!")
        return 1

    input_base = f"watermarked_ffmpeg_t9_crf30"
    input_video = input_base + INPUT_EXTENSION
    original_id: Optional[str] = None

    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r", encoding='utf-8') as f_id:
                original_id = f_id.read().strip()
            if not (original_id and len(original_id) == PAYLOAD_LEN_BYTES * 2):
                logging.error(f"ID в '{ORIGINAL_WATERMARK_FILE}' неверной длины.")
                original_id = None
            else:
                int(original_id, 16)
                logging.info(f"Original ID loaded: {original_id}")
        except ValueError:
            logging.error(f"ID в '{ORIGINAL_WATERMARK_FILE}' не hex.");
            original_id = None
        except Exception as e_read_id:
            logging.error(f"Ошибка чтения ID: {e_read_id}");
            original_id = None
    else:
        logging.warning(f"Файл ID '{ORIGINAL_WATERMARK_FILE}' не найден.")

    logging.info(f"--- Начало извлечения из файла: '{input_video}' ---")
    if not os.path.exists(input_video):
        logging.critical(f"Входной файл не найден: '{input_video}'.");
        print(f"ОШИБКА: Файл не найден: '{input_video}'");
        return 1

    # --- Расчет необходимого числа кадров для чтения ---
    payload_len_bits = PAYLOAD_LEN_BYTES * 8
    packet_len_if_ecc = payload_len_bits
    packet_len_if_raw = payload_len_bits
    ecc_possible_for_first_calc = False

    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is not None:
        try:
            if payload_len_bits <= BCH_CODE_OBJECT.k:
                packet_len_if_ecc = BCH_CODE_OBJECT.n
                ecc_possible_for_first_calc = True
        except Exception:
            pass

    current_expect_hybrid_ecc = globals().get('expect_hybrid_ecc', True)

    expect_hybrid_for_calc = current_expect_hybrid_ecc
    if expect_hybrid_for_calc and not ecc_possible_for_first_calc:
        expect_hybrid_for_calc = False

    max_possible_bits = 0
    if expect_hybrid_for_calc:
        max_possible_bits = packet_len_if_ecc + max(0, MAX_TOTAL_PACKETS_global - 1) * packet_len_if_raw
    else:
        current_packet_len_for_calc = packet_len_if_ecc if USE_ECC and ecc_possible_for_first_calc else packet_len_if_raw
        max_possible_bits = MAX_TOTAL_PACKETS_global * current_packet_len_for_calc

    if BITS_PER_PAIR <= 0: logging.critical("BITS_PER_PAIR <= 0!"); return 1
    pairs_needed_for_extract = ceil(max_possible_bits / BITS_PER_PAIR) if max_possible_bits > 0 else 0

    if pairs_needed_for_extract == 0:
        logging.error("Не требуется обрабатывать ни одной пары (согласно расчетам).")
        return 1

    num_frames_to_read = pairs_needed_for_extract * 2
    logging.info(
        f"Требуется обработать {pairs_needed_for_extract} пар, необходимо прочитать {num_frames_to_read} кадров.")

    read_start_time = time.time()
    logging.info(f"Чтение первых {num_frames_to_read} кадров с помощью OpenCV из '{input_video}'...")
    frames_for_extraction = read_required_frames_opencv(input_video, num_frames_to_read)
    read_time = time.time() - read_start_time

    if frames_for_extraction is None:
        logging.critical("Критическая ошибка при чтении необходимых кадров. Прерывание.")
        return 1

    actual_frames_read = len(frames_for_extraction)
    logging.info(f"Прочитано {actual_frames_read} кадров для извлечения за {read_time:.2f} сек.")

    if actual_frames_read < 2:
        logging.error(f"Прочитано менее 2 кадров ({actual_frames_read}). Невозможно извлечь ЦВЗ.")
        return 1
    if actual_frames_read < num_frames_to_read:
        logging.warning(f"Прочитано кадров ({actual_frames_read}) меньше, чем требовалось ({num_frames_to_read}).")

    # --- Вызов основной функции извлечения со списком кадров ---
    extracted_bytes = extract_watermark_from_video(
        frames=frames_for_extraction,
        nr=N_RINGS,
        nrtu=NUM_RINGS_TO_USE,
        bp=BITS_PER_PAIR,
        cps=CANDIDATE_POOL_SIZE,
        ec=EMBED_COMPONENT,
        expect_hybrid_ecc=current_expect_hybrid_ecc,
        max_expected_packets=MAX_TOTAL_PACKETS_global,
        ue=USE_ECC,
        bch_code=BCH_CODE_OBJECT,
        device=device,
        dtcwt_fwd=dtcwt_fwd,
        plb=PAYLOAD_LEN_BYTES,
        mw=MAX_WORKERS_EXTRACT
    )

    # --- Вывод результатов ---
    print(f"\n--- Extraction Results ---")
    extracted_hex: Optional[str] = None
    if extracted_bytes:
        if len(extracted_bytes) == PAYLOAD_LEN_BYTES:
            extracted_hex = extracted_bytes.hex()
            print(f"  Payload Length OK.")
            print(f"  Decoded ID (Hex): {extracted_hex}")
            logging.info(f"Decoded ID: {extracted_hex}")
        else:
            print(f"  ERROR: Decoded length mismatch! {len(extracted_bytes)}B != {PAYLOAD_LEN_BYTES}B.")
            logging.error(f"Decoded length mismatch!")
    else:
        print(f"  Extraction FAILED (No payload).")
        logging.error("Extraction failed/No payload.")

    final_match_status = False
    if original_id:
        print(f"  Original ID (Hex): {original_id}")
        if extracted_hex and extracted_hex == original_id:
            print("\n  >>> ID MATCH <<<")
            logging.info("ID MATCH.")
            final_match_status = True
        else:
            print("\n  >>> !!! ID MISMATCH or FAILED !!! <<<")
            logging.warning("ID MISMATCH or Extraction Failed.")
    else:
        print("\n  Original ID unavailable for comparison.")

    logging.info("--- Extraction Main Process Finished ---")
    total_main_time = time.time() - main_start_time
    logging.info(f"--- Total Extractor Time: {total_main_time:.2f} sec ---")
    print(f"\nExtraction finished. Log: {LOG_FILENAME}")

    return 0 if extracted_bytes is not None else 1


if __name__ == "__main__":
    # --- Настройка логирования ---
    if not logging.getLogger().handlers:
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                            format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    # --- Проверка зависимостей ---
    missing_libs_critical = []
    if not globals().get('PYAV_AVAILABLE', False) and 'av' not in sys.modules: missing_libs_critical.append(
        "PyAV (av)")  # Проверка, если флаг не определен
    if not globals().get('PYTORCH_WAVELETS_AVAILABLE', False): missing_libs_critical.append("pytorch_wavelets")
    if not globals().get('TORCH_DCT_AVAILABLE', False): missing_libs_critical.append("torch-dct")
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

    if globals().get('USE_ECC', False) and not globals().get('GALOIS_AVAILABLE', False):
        print("\nПРЕДУПРЕЖДЕНИЕ: ECC включен, но библиотека 'galois' не найдена/не работает.")
        logging.warning("ECC включен, но Galois недоступен.")

    # --- Профилирование ---
    DO_PROFILING = False
    profiler_instance = None
    if DO_PROFILING:
        if 'KERNPROF_VAR' not in os.environ and 'profile' not in globals() and 'cProfile' in sys.modules:  # Проверяем импорт cProfile
            profiler_instance = cProfile.Profile();
            profiler_instance.enable()
            print("cProfile профилирование включено.")
            logging.info("cProfile профилирование включено.")
        elif 'profile' in globals() and callable(globals()['profile']):
            print("line_profiler активен (через декоратор @profile). cProfile не будет запущен.")
            logging.info("line_profiler активен. cProfile не запущен.")

    final_exit_code = 1
    try:
        final_exit_code = main()
    except FileNotFoundError as e_fnf_main:
        print(f"\nОШИБКА: Файл не найден: {e_fnf_main}")
        logging.critical(f"FileNotFoundError в __main__: {e_fnf_main}", exc_info=True)
    except Exception as e_global_main:
        print(f"\nКРИТИЧЕСКАЯ НЕОБРАБОТАННАЯ ОШИБКА: {e_global_main}")
        logging.critical(f"Необработанная ошибка в __main__: {e_global_main}", exc_info=True)
    finally:
        if DO_PROFILING and profiler_instance is not None:
            profiler_instance.disable()
            logging.info("cProfile профилирование выключено.")
            stats_obj = pstats.Stats(profiler_instance).strip_dirs().sort_stats("cumulative")
            print("\n--- Статистика Профилирования (cProfile, Top 30) ---");
            stats_obj.print_stats(30)
            profile_prof_file = f"profile_extract_main_t{BCH_T if 'BCH_T' in globals() else 'X'}.prof"
            profile_txt_file = f"profile_extract_main_t{BCH_T if 'BCH_T' in globals() else 'X'}.txt"
            try:
                stats_obj.dump_stats(profile_prof_file)
                with open(profile_txt_file, 'w', encoding='utf-8') as f_pstats:
                    ps = pstats.Stats(profiler_instance, stream=f_pstats).strip_dirs().sort_stats('cumulative');
                    ps.print_stats()
                print(f"Статистика профилирования сохранена: {profile_prof_file}, {profile_txt_file}")
                logging.info(f"Статистика профилирования сохранена: {profile_prof_file}, {profile_txt_file}")
            except Exception as e_pstats_save:
                logging.error(f"Ошибка сохранения статистики: {e_pstats_save}")

        logging.info(f"Скрипт watermark_extractor.py завершен с кодом выхода {final_exit_code}.")
        print(f"\nСкрипт завершен с кодом выхода {final_exit_code}.")
        sys.exit(final_exit_code)
