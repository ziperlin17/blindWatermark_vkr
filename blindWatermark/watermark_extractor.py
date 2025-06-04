import cProfile
import concurrent
import gc
import math
import pstats
import shutil
import subprocess
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

from line_profiler import profile
import cProfile
import pstats
import torch
import torch.nn.functional as F
from av.video.reformatter import VideoReformatter
from galois import BCH
from line_profiler import profile

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse

    PYTORCH_WAVELETS_AVAILABLE = True
except ImportError:
    PYTORCH_WAVELETS_AVAILABLE = False


    class DTCWTForward:
        pass


    class DTCWTInverse:
        pass


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
from collections import Counter, defaultdict
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

    BCH_TYPE = galois.BCH;
    GALOIS_IMPORTED = True;
    logging.info("galois library imported.")
except ImportError:
    class BCH:
        pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.info("galois library not found.")
except Exception as import_err:
    class BCH:
        pass; BCH_TYPE = BCH; GALOIS_IMPORTED = False; logging.error(f"Galois import error: {import_err}",
                                                                     exc_info=True)

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
    _test_bch_ok = False;
    _test_decode_ok = False
    try:
        _test_m = BCH_M;
        _test_t = BCH_T;
        _test_n = (1 << _test_m) - 1;
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
            logging.error(f"Неизвестное k для t={_test_t}"); expected_k = -1

        if expected_k != -1 and hasattr(_test_bch_galois, 't') and hasattr(_test_bch_galois, 'k') \
                and _test_bch_galois.t == _test_t and _test_bch_galois.k == expected_k:
            logging.info(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) OK.")
            _test_bch_ok = True;
            BCH_CODE_OBJECT = _test_bch_galois
        else:
            logging.error(
                f"galois BCH init mismatch! Ожидалось: t={_test_t}, k={expected_k}. Получено: t=getattr(_test_bch_galois, 't', 'N/A'), k=getattr(_test_bch_galois, 'k', 'N/A').")

        if _test_bch_ok and BCH_CODE_OBJECT is not None:
            try:
                _n_bits = BCH_CODE_OBJECT.n;
                _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8);
                GF2 = galois.GF(2);
                _dummy_cw_vec = GF2(_dummy_cw_bits)
                _msg, _flips = BCH_CODE_OBJECT.decode(_dummy_cw_vec, errors=True)
                _test_decode_ok = (_flips is not None or _flips == 0);
                logging.info(f"galois: decode() test {'OK' if _test_decode_ok else 'failed'}.")
            except Exception as decode_err:
                logging.error(f"galois: decode() test failed: {decode_err}", exc_info=True); _test_decode_ok = False
    except Exception as test_err:
        logging.error(f"galois: ОШИБКА теста: {test_err}", exc_info=True); BCH_CODE_OBJECT = None; _test_bch_ok = False
    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None
if GALOIS_AVAILABLE:
    logging.info("galois: Готов к использованию.")
else:
    logging.warning("galois: НЕ ДОСТУПЕН.")

# --- Настройка логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                    format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование конфигурации ---
logging.info(f"--- Запуск Скрипта Извлечения (PyTorch Wavelets & DCT) ---")
logging.info(f"PyTorch Wavelets Доступно: {PYTORCH_WAVELETS_AVAILABLE}")
logging.info(f"Torch DCT Доступно: {TORCH_DCT_AVAILABLE}")
logging.info(f"Метод выбора колец: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}")
logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit")
logging.info(
    f"ECC Ожидается (для 1-го пак.): {USE_ECC}, Доступен/Работает: {GALOIS_AVAILABLE} (BCH m={BCH_M}, t={BCH_T})")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}, N_RINGS_Total={N_RINGS}")
logging.info(f"Параллелизм: ThreadPoolExecutor (max_workers={MAX_WORKERS_EXTRACT or 'default'}) с батчингом.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error(
    f"NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE! Проверьте настройки.")
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
        return s_values[0]  # Возвращаем тензор (скаляр)
    except Exception as e:
        logging.error(f"PyTorch SVD error: {e}", exc_info=True)
        return None


def dtcwt_pytorch_forward(yp_tensor: torch.Tensor, xfm: DTCWTForward, device: torch.device, fn: int = -1) -> Tuple[
    Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Применяет прямое DTCWT PyTorch к одному каналу (2D тензору)."""
    if not PYTORCH_WAVELETS_AVAILABLE: logging.error("PTW unavailable."); return None, None
    if not isinstance(yp_tensor, torch.Tensor) or yp_tensor.ndim != 2: logging.error(
        f"[F:{fn}] Invalid input tensor."); return None, None
    try:
        yp_tensor = yp_tensor.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        xfm = xfm.to(device)
        with torch.no_grad():
            Yl, Yh = xfm(yp_tensor)
        if Yl is None or Yh is None or not isinstance(Yh, list) or not Yh: logging.error(
            f"[F:{fn}] DTCWTForward invalid result."); return None, None
        return Yl, Yh
    except Exception as e:
        logging.error(f"[F:{fn}] PT DTCWT fwd error: {e}"); return None, None


def ring_division(lp_tensor: torch.Tensor, nr: int = N_RINGS, fn: int = -1) -> List[Optional[torch.Tensor]]:
    """Разбивает 2D PyTorch тензор на N колец (версия из embedder)."""
    if not isinstance(lp_tensor, torch.Tensor) or lp_tensor.ndim != 2: logging.error(
        f"[F:{fn}] Invalid input for ring_division."); return [None] * nr
    H, W = lp_tensor.shape;
    device = lp_tensor.device
    if H < 2 or W < 2: logging.warning(f"[F:{fn}] Tensor too small ({H}x{W})"); return [None] * nr
    try:
        rr, cc = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0;
        distances = torch.sqrt((rr - center_r) ** 2 + (cc - center_c) ** 2)
        min_dist, max_dist = torch.tensor(0.0, device=device), torch.max(distances)
        if max_dist < 1e-9:
            ring_bins = torch.tensor([0.0] * (nr + 1), device=device); ring_bins[1:] = max_dist + 1e-6
        else:
            ring_bins = torch.linspace(min_dist.item(), (max_dist + 1e-6).item(), nr + 1, device=device)
        ring_indices = torch.zeros_like(distances, dtype=torch.long) - 1
        for i in range(nr):
            mask = (distances >= ring_bins[i]) & (
                distances < ring_bins[i + 1] if i < nr - 1 else distances <= ring_bins[i + 1])
            ring_indices[mask] = i
        ring_indices[distances < ring_bins[1]] = 0
        rings: List[Optional[torch.Tensor]] = [None] * nr
        for rdx in range(nr):
            coords_tensor = torch.nonzero(ring_indices == rdx, as_tuple=False)
            if coords_tensor.shape[0] > 0: rings[rdx] = coords_tensor.long()
        return rings
    except Exception as e:
        logging.error(f"Ring division PT error F{fn}: {e}"); return [None] * nr


def calculate_entropies_torch(rv_tensor: torch.Tensor, fn: int = -1, ri: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет шенноновскую энтропию и "энтропию столкновений" (ваша формула)
    для одномерного PyTorch тензора значений пикселей (нормализованных).
    Все вычисления выполняются на устройстве rv_tensor.
    """
    eps = 1e-12
    shannon_entropy = torch.tensor(0.0, device=rv_tensor.device, dtype=rv_tensor.dtype)
    collision_entropy = torch.tensor(0.0, device=rv_tensor.device, dtype=rv_tensor.dtype)

    if rv_tensor.numel() > 0:
        if torch.all(rv_tensor == rv_tensor[0]):
            return shannon_entropy, collision_entropy

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
    seed_str = str(pi).encode('utf-8');
    hash_digest = hashlib.sha256(seed_str).digest()
    seed_int = int.from_bytes(hash_digest, 'big');
    prng = random.Random(seed_int)
    try:
        candidate_indices = prng.sample(range(nr), ps)
    except ValueError:
        candidate_indices = list(range(nr)); prng.shuffle(candidate_indices); candidate_indices = candidate_indices[:ps]
    logging.debug(f"[P:{pi}] Candidates: {candidate_indices}");
    return candidate_indices


def bits_to_bytes_strict(bit_list: List[int], expected_bytes: Optional[int] = None) -> Optional[bytes]:
    num_bits = len(bit_list)
    if not all(b in (0, 1) for b in bit_list):
        logging.error("bits_to_bytes_strict: список содержит невалидные биты (не 0 или 1).")
        return None
    if num_bits == 0 and expected_bytes == 0: return b''
    if num_bits == 0 and expected_bytes is None: return b''
    if num_bits == 0: return None  # Если ожидались байты, а бит нет

    if num_bits % 8 != 0:
        logging.error(f"bits_to_bytes_strict: длина списка бит ({num_bits}) не кратна 8.")
        return None

    actual_len_bytes = num_bits // 8
    if expected_bytes is not None and actual_len_bytes != expected_bytes:
        logging.error(
            f"bits_to_bytes_strict: фактическая длина байт ({actual_len_bytes}) не соответствует ожидаемой ({expected_bytes}).")
        return None

    byte_array = bytearray()
    for i in range(0, num_bits, 8):
        byte_chunk = bit_list[i:i + 8]
        try:
            byte_val = int("".join(map(str, byte_chunk)), 2)
            byte_array.append(byte_val)
        except ValueError:
            logging.error(f"bits_to_bytes_strict: Не удалось конвертировать битовый чанк: {byte_chunk}")
            return None
    return bytes(byte_array)


def get_byte_from_bits(bits_payload: List[int], byte_index: int) -> Optional[int]:
    start_bit_index = byte_index * 8
    end_bit_index = start_bit_index + 8
    if start_bit_index < 0 or end_bit_index > len(bits_payload):
        return None
    byte_bits_str = "".join(map(str, bits_payload[start_bit_index:end_bit_index]))
    try:
        return int(byte_bits_str, 2)
    except ValueError:
        return None


def get_bits_from_byte_value(byte_value: int) -> List[int]:
    if not (0 <= byte_value <= 255):
        raise ValueError("Значение байта должно быть в диапазоне 0-255")
    return [int(bit) for bit in format(byte_value, '08b')]


def intelligent_voting_strategy(  # Новое имя для ясности
        valid_packets_info: List[Dict[str, Any]],
        # {'payload_bits': List[int], 'packet_type': str, 'corrected_errors': int}
        payload_len_bytes: int,
        ngram_context_weight: float = 0.6  # Все еще нужен для Кандидата 3 (N-граммного)
) -> List[str]:
    payload_len_bits = payload_len_bytes * 8
    num_valid_packets = len(valid_packets_info)

    if num_valid_packets == 0:
        logging.error("Voting FinalVote: Нет валидных пакетов.")
        return []

    logging.info(f"Voting FinalVote: Голосование по {num_valid_packets} пакетам...")
    print(f"\n--- Voting FinalVote (по {num_valid_packets} пакетам) ---")

    # --- Этап 1: Формирование Кандидата 1 (побитовое, ничьи в 0) ---
    candidate1_bits: List[int] = [0] * payload_len_bits
    print(f"\n--- Формирование Кандидата 1 (побитовый) ---")
    for j_c1 in range(payload_len_bits):
        votes_0_c1, votes_1_c1 = 0, 0
        for p_info_c1 in valid_packets_info:
            if j_c1 < len(p_info_c1["payload_bits"]):
                if p_info_c1["payload_bits"][j_c1] == 1:
                    votes_1_c1 += 1
                elif p_info_c1["payload_bits"][j_c1] == 0:
                    votes_0_c1 += 1
        if votes_1_c1 > votes_0_c1:
            candidate1_bits[j_c1] = 1
        elif votes_0_c1 > votes_1_c1:
            candidate1_bits[j_c1] = 0
        else:
            candidate1_bits[j_c1] = 0  # Ничья -> 0
    candidate1_bytes = bits_to_bytes_strict(candidate1_bits, payload_len_bytes)
    candidate1_hex = candidate1_bytes.hex() if candidate1_bytes else None
    if not candidate1_hex: logging.error("К1: Ошибка HEX."); candidate1_bits = None  # Отмечаем C1 как невалидный

    # --- Этап 2: Формирование Кандидата 2 (простое побайтовое с приоритетом ECC) ---
    print(f"\n--- Формирование Кандидата 2 (простое побайтовое) ---")
    candidate2_bits: List[int] = [0] * payload_len_bits
    byte_counts_at_each_position: List[Counter] = [Counter() for _ in range(payload_len_bytes)]
    for byte_idx_c2 in range(payload_len_bytes):
        byte_options_map_c2: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for p_info_c2 in valid_packets_info:
            byte_val_c2 = get_byte_from_bits(p_info_c2["payload_bits"], byte_idx_c2)
            if byte_val_c2 is not None:
                byte_options_map_c2[byte_val_c2].append(p_info_c2)
                byte_counts_at_each_position[byte_idx_c2][byte_val_c2] += 1
        chosen_byte_c2_val: int
        if not byte_options_map_c2:
            fb_c1_byte = get_byte_from_bits(candidate1_bits, byte_idx_c2) if candidate1_bits else None
            chosen_byte_c2_val = fb_c1_byte if fb_c1_byte is not None else 0
        else:
            sorted_options_c2 = sorted(byte_options_map_c2.items(), key=lambda i: len(i[1]), reverse=True)
            max_freq_c2 = len(sorted_options_c2[0][1])
            winners_c2 = sorted([b for b, p in sorted_options_c2 if len(p) == max_freq_c2])
            if len(winners_c2) == 1:
                chosen_byte_c2_val = winners_c2[0]
            else:
                best_ecc_c2, min_corr_c2, clean_ecc_c2 = None, float('inf'), None
                for b_win in winners_c2:
                    for p_info in byte_options_map_c2[b_win]:
                        if p_info["packet_type"] == 'ECC':
                            corr = p_info.get("corrected_errors", 0)
                            if corr > 0 and corr < min_corr_c2:
                                min_corr_c2 = corr;best_ecc_c2 = b_win
                            elif corr == 0 and clean_ecc_c2 is None:
                                clean_ecc_c2 = b_win
                if best_ecc_c2 is not None:
                    chosen_byte_c2_val = best_ecc_c2
                elif clean_ecc_c2 is not None:
                    chosen_byte_c2_val = clean_ecc_c2
                else:
                    chosen_byte_c2_val = winners_c2[0]
        bits_c2 = get_bits_from_byte_value(chosen_byte_c2_val)
        for k in range(8): candidate2_bits[byte_idx_c2 * 8 + k] = bits_c2[k]
    candidate2_bytes = bits_to_bytes_strict(candidate2_bits, payload_len_bytes)
    candidate2_hex = candidate2_bytes.hex() if candidate2_bytes else None
    if not candidate2_hex: logging.error("К2: Ошибка HEX."); candidate2_bits = None

    # --- Этап 3: Формирование Кандидата 3 (N-граммный) ---
    print(f"\n--- Формирование Кандидата 3 (N-граммный) ---")
    candidate3_bits: List[int] = [0] * payload_len_bits
    bigram_counts: Counter = Counter()
    for p_info_bg in valid_packets_info:
        prev_b: Optional[int] = None
        for b_idx_bg in range(payload_len_bytes):
            curr_b = get_byte_from_bits(p_info_bg["payload_bits"], b_idx_bg)
            if curr_b is not None:
                if prev_b is not None: bigram_counts[(prev_b, curr_b)] += 1
                prev_b = curr_b
            else:
                prev_b = None

    temp_c3_bytes_list: List[int] = [0] * payload_len_bytes
    first_b_c3_init = get_byte_from_bits(candidate2_bits if candidate2_bits else candidate1_bits, 0)  # type: ignore
    temp_c3_bytes_list[0] = first_b_c3_init if first_b_c3_init is not None else 0

    for b_idx_c3 in range(1, payload_len_bytes):
        prev_b_c3 = temp_c3_bytes_list[b_idx_c3 - 1]
        opts_c3 = byte_counts_at_each_position[b_idx_c3]
        if not opts_c3:
            fb_c1_b = get_byte_from_bits(candidate1_bits, b_idx_c3) if candidate1_bits else None  # type: ignore
            temp_c3_bytes_list[b_idx_c3] = fb_c1_b if fb_c1_b is not None else 0
            continue
        best_b_c3: Optional[int] = None;
        max_s_c3 = -1.0
        count_prev_b_for_norm = byte_counts_at_each_position[b_idx_c3 - 1].get(prev_b_c3, 1)  # Знаменатель не 0
        if count_prev_b_for_norm == 0: count_prev_b_for_norm = 1.0  # type: ignore

        for cand_b, own_f in opts_c3.items():
            bigram_f = bigram_counts.get((prev_b_c3, cand_b), 0)
            s = (own_f / num_valid_packets) * (1.0 - ngram_context_weight) + (
                        bigram_f / count_prev_b_for_norm) * ngram_context_weight
            if s > max_s_c3:
                max_s_c3 = s; best_b_c3 = cand_b
            elif s == max_s_c3:
                if best_b_c3 is None or cand_b < best_b_c3: best_b_c3 = cand_b
        temp_c3_bytes_list[b_idx_c3] = best_b_c3 if best_b_c3 is not None else (
                    get_byte_from_bits(candidate1_bits, b_idx_c3) or 0)  # type: ignore

    for i_c3b, val_c3b in enumerate(temp_c3_bytes_list):
        bits_val_c3b = get_bits_from_byte_value(val_c3b)
        for k_c3b in range(8): candidate3_bits[i_c3b * 8 + k_c3b] = bits_val_c3b[k_c3b]
    candidate3_bytes = bits_to_bytes_strict(candidate3_bits, payload_len_bytes)
    candidate3_hex = candidate3_bytes.hex() if candidate3_bytes else None
    if not candidate3_hex: logging.error("К3: Ошибка HEX."); candidate3_bits = None

    logging.info(f"Кандидат 1 (побитовый): {candidate1_hex if candidate1_hex else 'Ошибка'}")
    logging.info(f"Кандидат 2 (побайтовый): {candidate2_hex if candidate2_hex else 'Ошибка'}")
    logging.info(f"Кандидат 3 (N-граммный): {candidate3_hex if candidate3_hex else 'Ошибка'}")

    # --- Этап 4: Финальное Побитовое Голосование по Трем Кандидатам ---
    final_resolved_bits: List[int] = [0] * payload_len_bits

    # Собираем только валидно сформированных кандидатов (в виде списков бит)
    valid_candidate_bit_lists: List[List[int]] = []
    if candidate1_bits: valid_candidate_bit_lists.append(candidate1_bits)
    if candidate2_bits: valid_candidate_bit_lists.append(candidate2_bits)
    if candidate3_bits: valid_candidate_bit_lists.append(candidate3_bits)

    if not valid_candidate_bit_lists:
        logging.error("Ни один из методов (C1, C2, C3) не смог сформировать битового кандидата.")
        return []  # Пустой список при фатальной ошибке

    print(f"\n--- Этап 4: Финальное Побитовое Голосование по {len(valid_candidate_bit_lists)} кандидатам ---")
    print(f"{'Bit Pos':<8} | {'Votes 0':<8} | {'Votes 1':<8} | {'Final Bit':<10} | {'Source':<15}")
    print("-" * (8 + 9 + 9 + 11 + 16))

    for j_final in range(payload_len_bits):
        votes_0_final, votes_1_final = 0, 0
        # Биты от кандидатов C3, C2, C1 для этой позиции j_final
        bit_from_c3: Optional[int] = candidate3_bits[j_final] if candidate3_bits and j_final < len(
            candidate3_bits) else None
        bit_from_c2: Optional[int] = candidate2_bits[j_final] if candidate2_bits and j_final < len(
            candidate2_bits) else None
        bit_from_c1: Optional[int] = candidate1_bits[j_final] if candidate1_bits and j_final < len(
            candidate1_bits) else None

        # Голосуем только по валидным кандидатам
        if bit_from_c1 is not None:
            if bit_from_c1 == 0:
                votes_0_final += 1
            else:
                votes_1_final += 1
        if bit_from_c2 is not None:
            if bit_from_c2 == 0:
                votes_0_final += 1
            else:
                votes_1_final += 1
        if bit_from_c3 is not None:
            if bit_from_c3 == 0:
                votes_0_final += 1
            else:
                votes_1_final += 1

        source_str = "Majority"
        if votes_1_final > votes_0_final:
            final_resolved_bits[j_final] = 1
        elif votes_0_final > votes_1_final:
            final_resolved_bits[j_final] = 0
        else:  # Ничья на финальном этапе (например, если был только 1 или 2 кандидата, или все 3 разные)
            logging.warning(
                f"Финальное голосование: Ничья на бите {j_final} ({votes_0_final}v{votes_1_final}). Приоритет C3>C2>C1.")
            if bit_from_c3 is not None:  # Приоритет C3 (N-граммный)
                final_resolved_bits[j_final] = bit_from_c3
                source_str = "Tie->C3(Ngram)"
            elif bit_from_c2 is not None:  # Затем C2 (Простой побайтовый)
                final_resolved_bits[j_final] = bit_from_c2
                source_str = "Tie->C2(Byte)"
            elif bit_from_c1 is not None:  # Затем C1 (Побитовый)
                final_resolved_bits[j_final] = bit_from_c1
                source_str = "Tie->C1(Bit)"
            else:  # Не должно случиться, если хоть один кандидат был
                final_resolved_bits[j_final] = 0  # Абсолютный дефолт
                source_str = "Tie->Default(0)"
                logging.error(f"  Бит {j_final}: Ничья не разрешена приоритетом! Установлен '0'.")
        print(
            f"{j_final:<8} | {votes_0_final:<8} | {votes_1_final:<8} | {final_resolved_bits[j_final]:<10} | {source_str:<15}")
    print("-" * (8 + 9 + 9 + 11 + 16))

    final_bytes_v4_obj = bits_to_bytes_strict(final_resolved_bits, payload_len_bytes)
    if not final_bytes_v4_obj:
        logging.error("Не удалось сформировать итоговый ID после финального голосования.")
        # Возвращаем лучший из C1, C2, C3, если они были. C1 должен быть всегда, если нет ранних ошибок.
        return [candidate1_hex] if candidate1_hex else []

    hex_candidate_v4_final = final_bytes_v4_obj.hex()
    final_hex_candidates_output = [hex_candidate_v4_final]
    logging.info(f"Итоговый кандидат после финального голосования (v4): {hex_candidate_v4_final}")

    # --- Формирование Второго Кандидата (упрощенная логика) ---
    # Основной результат - hex_candidate_v4_final.
    # Если он отличается от C1 (побитового) и C1 не совпадает с C3 (N-граммным),
    # то C1 может быть вторым кандидатом.
    # Или если V4 совпал с C1, но C3 был другим, то C3 может быть вторым.

    cand_to_compare_with_v4 = None
    if candidate1_hex and candidate1_hex != hex_candidate_v4_final:
        cand_to_compare_with_v4 = candidate1_hex
    elif candidate3_hex and candidate3_hex != hex_candidate_v4_final:  # Если C1 = V4, но C3 другой
        cand_to_compare_with_v4 = candidate3_hex
    # Можно еще добавить C2, если и C1 и C3 совпали с V4, но C2 был другим.

    if cand_to_compare_with_v4 and cand_to_compare_with_v4 not in final_hex_candidates_output:
        final_hex_candidates_output.append(cand_to_compare_with_v4)
        logging.info(f"Добавлен второй кандидат (из C1/C3): {cand_to_compare_with_v4}")

    # Гарантируем не более двух УНИКАЛЬНЫХ кандидатов
    unique_final_output = list(dict.fromkeys(final_hex_candidates_output))
    if len(unique_final_output) > 2:
        # Если получилось больше двух (маловероятно с текущей логикой), берем первые два
        unique_final_output = unique_final_output[:2]
        logging.warning(f"После всех шагов кандидатов >2, обрезано до: {unique_final_output}")

    print(f"\n--- Финально Выбранные Кандидаты (intelligent_voting_strategy_v4) ---")
    for i_res_v4, cand_hex_v4_val in enumerate(unique_final_output):
        print(f"Итоговый Кандидат {i_res_v4 + 1}: {cand_hex_v4_val}")

    return unique_final_output


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
        byte_chunk = valid_bits[i:i + 8]
        try:
            if len(byte_chunk) != 8: logging.error(f"Byte chunk error: {byte_chunk}"); return None
            byte_val = int("".join(map(str, byte_chunk)), 2);
            byte_array.append(byte_val)
        except ValueError:
            logging.error(f"Invalid symbols in bit chunk: {byte_chunk}"); return None
    return bytes(byte_array)


def decode_ecc(packet_bits_list: List[int], bch_code: Optional[BCH_TYPE], expected_data_len_bytes: int) -> Tuple[
    Optional[List[int]], int]:
    if not GALOIS_AVAILABLE or bch_code is None:
        return None, -1  # ECC недоступен

    n_corrected: int = -1  # По умолчанию - ошибка / неисправимо
    payload_len_bits = expected_data_len_bytes * 8

    try:
        if not (hasattr(bch_code, 'n') and hasattr(bch_code, 'k') and hasattr(bch_code, 'field') and hasattr(bch_code,
                                                                                                             'decode')):
            logging.error("decode_ecc: Объект bch_code не имеет необходимых атрибутов.")
            return None, -1

        n_val: int = bch_code.n
        k_val: int = bch_code.k

        if len(packet_bits_list) != n_val:
            logging.error(f"Decode ECC: Неверная длина входного пакета {len(packet_bits_list)} != {n_val}.")
            return None, -1
        if payload_len_bits > k_val:
            logging.error(f"Decode ECC: Ожидаемая длина полезной нагрузки ({payload_len_bits}) > k ({k_val}).")
            return None, -1

        packet_bits_np_arr = np.array(packet_bits_list, dtype=np.uint8)
        GF_field = bch_code.field
        rx_codeword_vector = GF_field(packet_bits_np_arr)

        try:
            corrected_msg_vector, num_errors_found = bch_code.decode(rx_codeword_vector, errors=True)
            n_corrected = int(num_errors_found)
        except galois.errors.UncorrectableError:
            logging.warning(
                f"Galois ECC: Неисправимые ошибки в пакете (длина {len(packet_bits_list)}). n_corrected остается -1.")
            # ВАШЕ ТРЕБОВАНИЕ: если -1, пакет все еще должен учитываться.
            # Значит, мы должны вернуть исходные биты (обрезанные до payload_len_bits), если это возможно.
            # Но decode_ecc в galois не возвращает их при UncorrectableError.
            # Поэтому, если ECC не смог исправить, мы не можем получить "исходные биты сообщения из кодового слова".
            # Мы можем вернуть только None для бит.
            return None, -1
        except Exception as e_decoding_error:
            logging.error(f"Decode ECC: Ошибка во время bch_code.decode: {e_decoding_error}", exc_info=True)
            return None, -1

        corrected_k_bits_np_arr = corrected_msg_vector.view(np.ndarray).astype(np.uint8)

        if corrected_k_bits_np_arr.size < payload_len_bits:
            logging.error(
                f"Decode ECC: Длина декодированных бит ({corrected_k_bits_np_arr.size}) < ожидаемой ({payload_len_bits}).")
            return None, n_corrected

        final_payload_bits_list = corrected_k_bits_np_arr[:payload_len_bits].tolist()

        # logging.info(f"Galois ECC: Пакет успешно декодирован, исправлено ошибок: {n_corrected}.")
        return final_payload_bits_list, n_corrected

    except Exception as e_general_ecc:
        logging.error(f"Decode ECC: Неожиданная ошибка: {e_general_ecc}", exc_info=True)
        return None, -1


@profile
def extract_single_bit(L1_tensor: torch.Tensor, L2_tensor: torch.Tensor, ring_idx: int, n_rings: int, fn: int) -> \
Optional[int]:
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
                or not (0 <= ring_idx < n_rings and ring_idx < len(r1c) and ring_idx < len(r2c)):
            logging.warning(f"{prefix} Invalid ring index or ring_division failed.")
            return None
        cd1_tensor = r1c[ring_idx];
        cd2_tensor = r2c[ring_idx]
        min_ring_size = 10
        if cd1_tensor is None or cd2_tensor is None or cd1_tensor.shape[0] < min_ring_size or cd2_tensor.shape[
            0] < min_ring_size:
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
                rv1_tensor = rv1_tensor[:min_s];
                rv2_tensor = rv2_tensor[:min_s]

            logging.debug(
                f"{prefix} rv1 stats: size={rv1_tensor.numel()}, mean={rv1_tensor.mean():.6e}, std={rv1_tensor.std():.6e}")
            logging.debug(
                f"{prefix} rv2 stats: size={rv2_tensor.numel()}, mean={rv2_tensor.mean():.6e}, std={rv2_tensor.std():.6e}")

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
            logging.error(f"{prefix} PyTorch runtime error during Tensor DCT/SVD: {torch_err}", exc_info=True);
            return None
        except IndexError:
            logging.warning(f"{prefix} Index error getting ring tensor values.");
            return None
        except Exception as e:
            logging.error(f"{prefix} Error in Tensor DCT/SVD processing part: {e}", exc_info=True);
            return None

        # --- Шаг 4: Принятие решения ---
        eps = 1e-12;
        threshold = 1.0
        if abs(s2) < eps:
            logging.warning(f"{prefix} s2={s2:.2e} is close to zero. Unreliable ratio.")
            return None

        ratio = s1 / s2
        # --- Явное вычисление и логирование сравнения ---
        comparison_result = (ratio >= threshold)
        extracted_bit = 0 if comparison_result else 1

        logging.info(
            f"{prefix} Decision: s1={s1:.8e}, s2={s2:.8e}, ratio={ratio:.8f}, threshold={threshold}, comparison (ratio >= threshold)={comparison_result}, extracted_bit={extracted_bit}")

        return extracted_bit

    except Exception as e:
        logging.error(f"Unexpected error in extract_single_bit (P:{pair_index}, R:{ring_idx}): {e}", exc_info=True)
        return None


def _extract_batch_worker(batch_args_list: List[Dict]) -> Dict[int, List[Optional[int]]]:
    """
    Обрабатывает батч задач извлечения: выполняет DTCWT один раз на пару,
    выбирает кольца с использованием GPU-энтропии,
    затем вызывает extract_single_bit для выбранных колец.
    """
    batch_results: Dict[int, List[Optional[int]]] = {}
    if not batch_args_list:
        return {}

    args_example = batch_args_list[0]
    nr = args_example.get('n_rings', N_RINGS)
    # nrtu_effective - это BITS_PER_PAIR, т.е. сколько бит мы ожидаем извлечь из пары
    nrtu_effective = args_example.get('num_rings_to_use', NUM_RINGS_TO_USE)
    cps = args_example.get('candidate_pool_size', CANDIDATE_POOL_SIZE)
    ec = args_example.get('embed_component', EMBED_COMPONENT)
    device = args_example.get('device')
    dtcwt_fwd = args_example.get('dtcwt_fwd')

    if device is None or dtcwt_fwd is None:
        logging.error("Device или DTCWTForward не переданы в _extract_batch_worker (extractor)!")
        for args_item in batch_args_list:
            pair_idx_item = args_item.get('pair_idx', -1)
            if pair_idx_item != -1:
                batch_results[pair_idx_item] = [None] * nrtu_effective  # Заполняем None
        return batch_results

    for args in batch_args_list:
        pair_idx = args.get('pair_idx', -1)
        f1_bgr = args.get('frame1')
        f2_bgr = args.get('frame2')

        # Инициализируем результат для этой пары списком None нужной длины
        extracted_bits_for_pair: List[Optional[int]] = [None] * nrtu_effective

        if pair_idx == -1 or f1_bgr is None or f2_bgr is None:
            logging.error(
                f"Extractor: Недостаточно аргументов для обработки pair_idx={pair_idx if pair_idx != -1 else 'unknown'}")
            if pair_idx != -1: batch_results[pair_idx] = extracted_bits_for_pair  # Сохраняем None результат
            continue

        fn = 2 * pair_idx  # Номер первого кадра в паре
        L1_tensor: Optional[torch.Tensor] = None
        L2_tensor: Optional[torch.Tensor] = None

        try:
            if not isinstance(f1_bgr, np.ndarray) or not isinstance(f2_bgr, np.ndarray):
                logging.warning(f"[Extractor Worker P:{pair_idx}] Input frames not numpy arrays.")
                batch_results[pair_idx] = extracted_bits_for_pair;
                continue

            y1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2YCrCb)
            y2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2YCrCb)
            c1_np = y1[:, :, ec].copy().astype(np.float32) / 255.0
            c2_np = y2[:, :, ec].copy().astype(np.float32) / 255.0
            comp1_tensor = torch.from_numpy(c1_np).to(device=device)
            comp2_tensor = torch.from_numpy(c2_np).to(device=device)

            Yl_t, _ = dtcwt_pytorch_forward(comp1_tensor, dtcwt_fwd, device, fn)
            Yl_t1, _ = dtcwt_pytorch_forward(comp2_tensor, dtcwt_fwd, device, fn + 1)

            if Yl_t is None or Yl_t1 is None:
                logging.warning(f"[Extractor Worker P:{pair_idx}] DTCWT forward failed.")
                batch_results[pair_idx] = extracted_bits_for_pair;
                continue

            if Yl_t.dim() > 2:
                L1_tensor = Yl_t.squeeze(0).squeeze(0)
            elif Yl_t.dim() == 2:
                L1_tensor = Yl_t
            else:
                raise ValueError(f"Invalid Yl_t dim: {Yl_t.dim()}")

            if Yl_t1.dim() > 2:
                L2_tensor = Yl_t1.squeeze(0).squeeze(0)
            elif Yl_t1.dim() == 2:
                L2_tensor = Yl_t1
            else:
                raise ValueError(f"Invalid Yl_t1 dim: {Yl_t1.dim()}")

            if not torch.is_floating_point(L1_tensor) or not torch.is_floating_point(L2_tensor):
                raise TypeError(f"L1/L2 not float! L1:{L1_tensor.dtype}, L2:{L2_tensor.dtype}")
            if L1_tensor.shape != L2_tensor.shape:
                raise ValueError(f"L1/L2 shape mismatch! L1:{L1_tensor.shape}, L2:{L2_tensor.shape}")

            # --- Выбор колец с GPU энтропией ---
            # ring_division уже должна быть GPU-версией, возвращающей список тензоров координат
            coords_list_gpu = ring_division(L1_tensor, nr, fn)
            if coords_list_gpu is None or len(coords_list_gpu) != nr:
                logging.warning(f"[Extractor Worker P:{pair_idx}] Ring division failed.")
                batch_results[pair_idx] = extracted_bits_for_pair;
                continue

            candidate_rings = get_fixed_pseudo_random_rings(pair_idx, nr, cps)

            # Количество колец, которые мы пытаемся выбрать, равно nrtu_effective (BITS_PER_PAIR)
            num_rings_to_attempt_select = min(nrtu_effective, len(candidate_rings))

            if len(candidate_rings) < nrtu_effective:
                logging.warning(
                    f"[Extractor Worker P:{pair_idx}] Not enough candidates {len(candidate_rings)}<{nrtu_effective}. Will select {len(candidate_rings)}.")

            if num_rings_to_attempt_select == 0 and nrtu_effective > 0:
                logging.error(
                    f"[Extractor Worker P:{pair_idx}] No candidate rings to select from for {nrtu_effective} bits.")
                batch_results[pair_idx] = [None] * nrtu_effective
                continue

            entropies_gpu = []  # Список для (тензор_энтропии, индекс_кольца)
            min_pixels_for_entropy = 10

            # L1_tensor (LL-поддиапазон первого кадра) используется для выбора колец
            for r_idx_candidate in candidate_rings:
                entropy_val_tensor = torch.tensor(-float('inf'), device=device, dtype=L1_tensor.dtype)
                if 0 <= r_idx_candidate < len(coords_list_gpu) and \
                        coords_list_gpu[r_idx_candidate] is not None and \
                        coords_list_gpu[r_idx_candidate].shape[0] >= min_pixels_for_entropy:

                    coords_tensor_curr_ring = coords_list_gpu[r_idx_candidate]
                    try:
                        rows_curr_ring, cols_curr_ring = coords_tensor_curr_ring[:, 0], coords_tensor_curr_ring[:, 1]
                        # Извлекаем значения пикселей кольца из L1_tensor (уже на GPU)
                        rv_tensor_curr_ring = L1_tensor[rows_curr_ring, cols_curr_ring]
                        # Вычисляем Шенноновскую энтропию с помощью calculate_entropies_torch
                        shannon_entropy_tensor, _ = calculate_entropies_torch(rv_tensor_curr_ring.float(), fn,
                                                                              r_idx_candidate)
                        if torch.isfinite(shannon_entropy_tensor):
                            entropy_val_tensor = shannon_entropy_tensor
                    except IndexError:
                        logging.warning(
                            f"[Extractor P:{pair_idx} R_cand:{r_idx_candidate}] IndexError during GPU entropy.")
                    except Exception as e_entr_gpu:
                        logging.warning(
                            f"[Extractor P:{pair_idx} R_cand:{r_idx_candidate}] GPU Entropy calc error: {e_entr_gpu}")
                entropies_gpu.append((entropy_val_tensor, r_idx_candidate))

            # Сортировка по значению тензора энтропии
            entropies_gpu.sort(key=lambda x: x[0].item(), reverse=True)

            # Выбираем лучшие num_rings_to_attempt_select колец
            selected_rings = [idx for e_tensor, idx in entropies_gpu if e_tensor.item() > -float('inf')][
                             :num_rings_to_attempt_select]

            # Fallback, если после фильтрации по -inf не хватило до num_rings_to_attempt_select
            if len(selected_rings) < num_rings_to_attempt_select:
                logging.warning(
                    f"[Extractor Worker P:{pair_idx}] Fallback ring selection after GPU entropy ({len(selected_rings)}<{num_rings_to_attempt_select}).")
                for r_cand in candidate_rings:  # Идем по исходным кандидатам
                    if len(selected_rings) >= num_rings_to_attempt_select: break
                    if r_cand not in selected_rings: selected_rings.append(r_cand)

            # Если итоговое количество выбранных колец меньше, чем мы ожидали извлечь бит (nrtu_effective),
            # то для недостающих бит результат будет None.
            actual_rings_for_extraction = selected_rings  # Это те кольца, из которых будем реально извлекать

            logging.info(
                f"[Extractor Worker P:{pair_idx}] Selected {len(actual_rings_for_extraction)} rings using GPU entropy: {actual_rings_for_extraction}. Expecting to fill {nrtu_effective} bit slots.")

            # extracted_bits_for_pair уже инициализирован как [None] * nrtu_effective
            for i, ring_idx_to_extract in enumerate(actual_rings_for_extraction):
                # Убедимся, что не выходим за пределы nrtu_effective, если колец оказалось больше (не должно, но для безопасности)
                if i < nrtu_effective:
                    extracted_bits_for_pair[i] = extract_single_bit(L1_tensor, L2_tensor, ring_idx_to_extract, nr, fn)

            batch_results[
                pair_idx] = extracted_bits_for_pair  # Содержит nrtu_effective элементов, некоторые могут быть None

        except cv2.error as cv_err:
            logging.error(f"OpenCV error P:{pair_idx} in Extractor worker: {cv_err}", exc_info=True)
            if pair_idx != -1: batch_results[pair_idx] = [None] * nrtu_effective
        except RuntimeError as torch_err:
            logging.error(f"PyTorch runtime error P:{pair_idx} in Extractor worker: {torch_err}", exc_info=True)
            if pair_idx != -1: batch_results[pair_idx] = [None] * nrtu_effective
        except Exception as e:
            logging.error(f"Unexpected error processing pair {pair_idx} in Extractor worker: {e}", exc_info=True)
            if pair_idx != -1: batch_results[pair_idx] = [None] * nrtu_effective

    return batch_results


# --- Вспомогательная функция чтения кадров ---
def read_required_frames_pyav(
        video_path: str,
        num_frames_to_read: int,
        preferred_video_stream_index: Optional[int] = None
) -> Optional[List[np.ndarray]]:
    """
    Читает ТОЛЬКО первые num_frames_to_read видеокадров из указанного видеопотока
    с помощью PyAV и конвертирует их в список BGR NumPy массивов.

    Args:
        video_path: Путь к видеофайлу.
        num_frames_to_read: Количество видеокадров для чтения.
        preferred_video_stream_index: Предпочтительный индекс видеопотока.
                                      Если None или не найден, будет выбран первый видеопоток.

    Returns:
        Список NumPy массивов (кадры в BGR) или None в случае критической ошибки.
        Может вернуть пустой список, если num_frames_to_read <= 0.
    """
    # Предполагается, что PYAV_AVAILABLE - это глобальная переменная, как в вашем коде
    if not PYAV_AVAILABLE:
        logging.error("PyAV недоступен для read_required_frames_pyav.")
        return None

    if num_frames_to_read <= 0:
        logging.debug("read_required_frames_pyav: num_frames_to_read <= 0, возвращаем пустой список.")
        return []

    frames_bgr_list: List[np.ndarray] = []
    container: Optional[av.container.Container] = None
    frames_decoded_count = 0

    reformatter_to_bgr: Optional[VideoReformatter] = None
    reformatter_yuv_for_cv_fallback: Optional[VideoReformatter] = None  # Переименовал для ясности

    logging.info(f"[PyAV Read Limited] Открытие: '{video_path}' для чтения {num_frames_to_read} кадров.")

    try:
        container = av.open(video_path, mode='r', metadata_errors='ignore')

        target_stream: Optional[av.stream.Stream] = None
        if not container.streams.video:
            logging.error(f"Видеопотоки не найдены в '{video_path}'.")
            if container: container.close()
            return None

        # Логика выбора видеопотока
        if preferred_video_stream_index is not None:
            if 0 <= preferred_video_stream_index < len(container.streams.video):
                target_stream = container.streams.video[preferred_video_stream_index]
                logging.info(f"Выбран указанный видеопоток с индексом: {target_stream.index}")
            else:
                logging.warning(f"Указанный индекс видеопотока {preferred_video_stream_index} некорректен "
                                f"(доступно {len(container.streams.video)}). Попытка использовать первый видеопоток.")
                target_stream = None  # Сброс для выбора первого потока ниже

        if target_stream is None:  # Если не был выбран предпочтительный или он был некорректен
            target_stream = container.streams.video[0]
            logging.info(
                f"Используется первый найденный видеопоток с индексом: {target_stream.index} (Codec: {target_stream.codec_context.name if target_stream.codec_context else 'N/A'})")

        if not target_stream.codec_context:  # Дополнительная проверка
            logging.error(f"У выбранного видеопотока (индекс {target_stream.index}) отсутствует codec_context.")
            if container: container.close()
            return None

        stream_fps_estimate_num = target_stream.average_rate or target_stream.r_frame_rate or 0
        stream_fps_estimate = float(stream_fps_estimate_num) if stream_fps_estimate_num else 0.0
        logging.debug(f"Чтение из потока: {target_stream.codec_context.name}, "
                      f"{target_stream.codec_context.width}x{target_stream.codec_context.height} @ "
                      f"{stream_fps_estimate:.2f} FPS (оценка)")

        for packet in container.demux(target_stream):
            if packet.dts is None: continue
            if packet.stream.index != target_stream.index: continue  # На всякий случай

            try:
                for frame in packet.decode():
                    if frames_decoded_count >= num_frames_to_read: break
                    if not (frame and isinstance(frame, av.VideoFrame)): continue

                    np_frame_bgr: Optional[np.ndarray] = None
                    frame_format_name = frame.format.name if frame.format else "unknown_format"

                    try:
                        np_frame_bgr = frame.to_ndarray(format='bgr24')
                    except (av.FFmpegError, ValueError, TypeError) as e_to_ndarray:
                        logging.debug(
                            f"PyAV: Не удалось напрямую конвертировать кадр {frames_decoded_count} (формат: {frame_format_name}) в bgr24: {e_to_ndarray}. Попытка реформатирования.")

                        if frame.width > 0 and frame.height > 0 and frame.format:
                            try:
                                if reformatter_to_bgr is None or \
                                        reformatter_to_bgr.width != frame.width or \
                                        reformatter_to_bgr.height != frame.height or \
                                        reformatter_to_bgr.src_format != frame.format.name:
                                    reformatter_to_bgr = VideoReformatter(frame.width, frame.height, 'bgr24',
                                                                          src_format=frame.format.name)
                                reformatted_frame_bgr = reformatter_to_bgr.reformat(frame)
                                np_frame_bgr = reformatted_frame_bgr.to_ndarray(format='bgr24')
                            except Exception as e_reformat_pyav_bgr:
                                logging.debug(
                                    f"PyAV: Ошибка реформатирования в BGR24 для кадра {frames_decoded_count}: {e_reformat_pyav_bgr}. Попытка через YUV+OpenCV.")
                                np_frame_bgr = None

                        if np_frame_bgr is None and frame.width > 0 and frame.height > 0 and frame.format:  # Если BGR реформат не удался
                            try:
                                if reformatter_yuv_for_cv_fallback is None or \
                                        reformatter_yuv_for_cv_fallback.width != frame.width or \
                                        reformatter_yuv_for_cv_fallback.height != frame.height or \
                                        reformatter_yuv_for_cv_fallback.src_format != frame.format.name:
                                    reformatter_yuv_for_cv_fallback = VideoReformatter(frame.width, frame.height,
                                                                                       'yuv420p',
                                                                                       src_format=frame.format.name)

                                frame_yuv = reformatter_yuv_for_cv_fallback.reformat(frame)
                                np_frame_yuv = frame_yuv.to_ndarray()

                                if np_frame_yuv.shape[0] * 2 // 3 == frame_yuv.height:
                                    np_frame_bgr = cv2.cvtColor(np_frame_yuv, cv2.COLOR_YUV2BGR_I420)
                                else:
                                    logging.warning(
                                        f"PyAV: Некорректная форма YUV массива ({np_frame_yuv.shape}) для кадра {frames_decoded_count} после реформатирования в yuv420p.")
                                    continue
                            except Exception as e_reformat_cv_fallback:
                                logging.error(
                                    f"PyAV: Ошибка fallback реформатирования (YUV+CV2) для кадра {frames_decoded_count}: {e_reformat_cv_fallback}",
                                    exc_info=False)
                                continue

                    if np_frame_bgr is not None:
                        frames_bgr_list.append(np_frame_bgr)
                        frames_decoded_count += 1
                    else:
                        logging.warning(
                            f"PyAV: Не удалось получить BGR NumPy для кадра {frames_decoded_count} (исходный формат: {frame_format_name}). Кадр пропущен.")
                        continue

                    if frames_decoded_count % 100 == 0 and frames_decoded_count > 0:
                        logging.debug(
                            f"[PyAV Read Limited] Прочитано и декодировано кадров: {frames_decoded_count}/{num_frames_to_read}")

            except (av.FFmpegError, ValueError) as e_decode:
                logging.warning(f"PyAV: Ошибка декодирования пакета (кадр ~{frames_decoded_count}): {e_decode}")
            except Exception as e_general_decode:
                logging.error(
                    f"PyAV: Неожиданная ошибка при декодировании пакета (кадр ~{frames_decoded_count}): {e_general_decode}",
                    exc_info=True)

            if frames_decoded_count >= num_frames_to_read: break

        logging.info(f"[PyAV Read Limited] Чтение завершено. Получено кадров: {len(frames_bgr_list)}.")

        if len(frames_bgr_list) < num_frames_to_read and num_frames_to_read > 0 and frames_decoded_count < num_frames_to_read:
            logging.warning(
                f"[PyAV Read Limited] Прочитано ({len(frames_bgr_list)}) меньше кадров, чем запрошено ({num_frames_to_read}). Возможно, достигнут конец файла раньше.")

        if len(frames_bgr_list) == 0 and num_frames_to_read > 0:
            logging.error(f"[PyAV Read Limited] Не удалось прочитать ни одного кадра из '{video_path}'.")
            return None

        return frames_bgr_list

    except av.FFmpegError as e_av_critical:
        logging.error(
            f"[PyAV Read Limited] Критическая ошибка PyAV/FFmpeg при открытии/чтении '{video_path}': {e_av_critical}",
            exc_info=True)
        return None
    except Exception as e_main_read:
        logging.error(f"[PyAV Read Limited] Неожиданная критическая ошибка при чтении '{video_path}': {e_main_read}",
                      exc_info=True)
        return None
    finally:
        if container:
            try:
                container.close()
                logging.debug("[PyAV Read Limited] Контейнер PyAV закрыт.")
            except Exception:
                logging.debug(
                    "[PyAV Read Limited] Ошибка при явном закрытии контейнера PyAV (возможно, уже был закрыт или ошибка при открытии).")


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
        nr: int,
        nrtu: int,
        bp: int,
        cps: int,
        ec: int,
        expect_hybrid_ecc: bool,
        max_expected_packets: int,
        ue: bool,
        bch_code: Optional[BCH_TYPE],
        device: Optional[torch.device],
        dtcwt_fwd: Optional[DTCWTForward],
        plb: int,
        mw: Optional[int],
        # Добавляем параметр для ngram_context_weight, если хотим его настраивать извне
        ngram_weight_param: float = 0.6  # Значение по умолчанию
) -> List[str]:  # ИЗМЕНЕНО: Возвращает список HEX-строк или пустой список
    """
    Основная функция извлечения ЦВЗ из предоставленного списка кадров.
    Использует ThreadPoolExecutor и интеллектуальную стратегию голосования.
    """
    # --- Проверки входных данных и доступности библиотек ---
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        logging.critical("Extract Watermark: Отсутствуют PyTorch Wavelets или Torch DCT!")
        return []
    if not frames:
        logging.error("Extract Watermark: Список кадров пуст! Нечего извлекать.")
        return []
    if device is None or dtcwt_fwd is None:
        logging.critical("Extract Watermark: Device или DTCWTForward не переданы!")
        return []

    if ue and expect_hybrid_ecc and not GALOIS_AVAILABLE:
        logging.warning(
            "Extract Watermark: ECC требуется для гибридного режима, но Galois недоступен! Это повлияет на декодирование.")

    logging.info(f"--- Extract Watermark: Запуск Извлечения (список {len(frames)} кадров, Intelligent Voting) ---")
    logging.info(
        f"Параметры: Hybrid={expect_hybrid_ecc}, MaxPkts={max_expected_packets}, NRTU={nrtu}, BP={bp}, NgramW={ngram_weight_param}")
    start_time = time.time()

    nf = len(frames)
    total_pairs_available = nf // 2
    if total_pairs_available == 0:
        logging.error("Extract Watermark: В предоставленном списке нет пар кадров для обработки.")
        return []

    payload_len_bits = plb * 8
    message_len_for_ecc = payload_len_bits
    actual_codeword_len_if_ecc = message_len_for_ecc
    packet_len_if_raw = payload_len_bits
    ecc_possible_for_first = False

    if ue and GALOIS_AVAILABLE and bch_code is not None and isinstance(bch_code, BCH_TYPE):
        try:
            if hasattr(bch_code, 'k') and hasattr(bch_code, 'n') and message_len_for_ecc <= bch_code.k:
                actual_codeword_len_if_ecc = bch_code.n
                ecc_possible_for_first = True
                logging.info(
                    f"ECC проверка: Возможно для 1-го пакета (n={bch_code.n}, k={bch_code.k}, t={bch_code.t}).")
            else:
                k_val = bch_code.k if hasattr(bch_code, 'k') else 'N/A'
                logging.warning(f"ECC проверка: Payload ({message_len_for_ecc}) > k ({k_val}). ECC не будет применен.")
        except Exception as e_galois_check:
            logging.error(f"ECC проверка: Ошибка параметров Galois: {e_galois_check}.")
    else:
        logging.info("ECC проверка: Либо USE_ECC=False, либо Galois недоступен/некорректен.")

    effective_expect_hybrid_ecc = expect_hybrid_ecc
    if effective_expect_hybrid_ecc and not ecc_possible_for_first:
        logging.warning(
            "Гибридный режим запрошен, но ECC для первого пакета невозможен/не настроен. Все пакеты будут обрабатываться как Raw.")
        effective_expect_hybrid_ecc = False

    max_possible_bits_to_extract = 0
    if effective_expect_hybrid_ecc:
        max_possible_bits_to_extract = actual_codeword_len_if_ecc + max(0, max_expected_packets - 1) * packet_len_if_raw
    else:
        len_of_each_packet = actual_codeword_len_if_ecc if ue and ecc_possible_for_first and not effective_expect_hybrid_ecc else packet_len_if_raw
        max_possible_bits_to_extract = max_expected_packets * len_of_each_packet

    if bp <= 0:
        logging.error(f"Bits per pair (bp) должен быть > 0, получено: {bp}")
        return []

    pairs_needed = ceil(max_possible_bits_to_extract / bp) if max_possible_bits_to_extract > 0 else 0
    pairs_to_process = min(total_pairs_available, pairs_needed)

    logging.info(f"Цель извлечения: до {max_expected_packets} пакетов (~{max_possible_bits_to_extract} бит).")
    logging.info(
        f"Пар кадров: Доступно={total_pairs_available}, Нужно={pairs_needed}, Будет обработано={pairs_to_process}")

    if pairs_to_process == 0:
        logging.warning("Нечего обрабатывать (pairs_to_process=0).")
        return []

    all_pairs_args: List[Dict[str, Any]] = []
    for pair_idx_prep in range(pairs_to_process):
        i1, i2 = 2 * pair_idx_prep, 2 * pair_idx_prep + 1
        if i2 >= nf:
            logging.warning(f"Недостаточно кадров для пары {pair_idx_prep}.")
            break
        if frames[i1] is None or frames[i2] is None:
            logging.warning(f"Пропуск пары {pair_idx_prep}: один из кадров None.")
            continue
        args = {'pair_idx': pair_idx_prep, 'frame1': frames[i1].copy(), 'frame2': frames[i2].copy(),
                'n_rings': nr, 'num_rings_to_use': nrtu, 'candidate_pool_size': cps,
                'embed_component': ec, 'device': device, 'dtcwt_fwd': dtcwt_fwd}
        all_pairs_args.append(args)

    num_valid_tasks = len(all_pairs_args)
    if num_valid_tasks == 0:
        logging.error("Нет валидных задач для ThreadPoolExecutor.")
        return []
    if num_valid_tasks < pairs_to_process:
        logging.info(f"Задач ({num_valid_tasks}) < планировалось ({pairs_to_process}). Обновляем pairs_to_process.")
        pairs_to_process = num_valid_tasks

    if pairs_to_process == 0:
        logging.warning("Не осталось пар для обработки после фильтрации.")
        return []

    num_workers = mw if mw is not None and mw > 0 else (os.cpu_count() or 1)
    num_workers = min(num_workers, num_valid_tasks)
    batch_size_calc = num_valid_tasks / (num_workers * 2) if num_workers > 0 else num_valid_tasks
    batch_size = max(1, ceil(batch_size_calc))

    batched_args_list: List[List[Dict[str, Any]]] = []
    if all_pairs_args:
        batched_args_list = [all_pairs_args[i: i + batch_size] for i in range(0, num_valid_tasks, batch_size) if
                             all_pairs_args[i:i + batch_size]]

    extracted_bits_map: Dict[int, List[Optional[int]]] = {}
    if batched_args_list:
        logging.info(
            f"Запуск {len(batched_args_list)} батчей ({num_valid_tasks} пар) в ThreadPoolExecutor (mw={num_workers}, batch_size≈{batch_size})...")
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_batch_info = {
                    executor.submit(_extract_batch_worker, batch_arg_list): {  # type: ignore
                        'start_pair_idx': batch_arg_list[0]['pair_idx'] if batch_arg_list else -1,
                        'num_pairs_in_batch': len(batch_arg_list)
                    } for batch_arg_list in batched_args_list
                }
                for future in concurrent.futures.as_completed(future_to_batch_info):
                    batch_info = future_to_batch_info[future]
                    try:
                        batch_results_map = future.result()
                        if batch_results_map: extracted_bits_map.update(batch_results_map)
                    except Exception as e_future:
                        logging.error(
                            f"Ошибка выполнения батча (начинающегося с пары ~{batch_info['start_pair_idx']}, {batch_info['num_pairs_in_batch']} пар): {e_future}",
                            exc_info=True)
        except Exception as e_executor:
            logging.critical(f"Критическая ошибка ThreadPoolExecutor: {e_executor}", exc_info=True)
            return []
    else:
        logging.warning("Нет батчей для обработки в ThreadPoolExecutor.")

    if not extracted_bits_map and pairs_to_process > 0:
        logging.error("Ни одной пары не было успешно обработано (карта результатов извлеченных бит пуста).")
        return []

    extracted_bits_all: List[Optional[int]] = []
    for pair_idx_loop in range(pairs_to_process):
        bits_from_pair = extracted_bits_map.get(pair_idx_loop)
        if bits_from_pair and isinstance(bits_from_pair, list) and len(bits_from_pair) == bp:
            extracted_bits_all.extend(bits_from_pair)
        else:
            extracted_bits_all.extend([None] * bp)

    valid_bits_for_decoding = [b for b in extracted_bits_all if b is not None and b in (0, 1)]
    logging.info(
        f"Сборка бит: Всего извлечено (с None): {len(extracted_bits_all)}, Валидных (0/1): {len(valid_bits_for_decoding)}.")
    if not valid_bits_for_decoding:
        logging.error("Нет валидных бит (0/1) для декодирования.")
        return []

        # --- Формирование all_packets_details для intelligent_voting_strategy ---
    all_packets_details: List[Dict[str, Any]] = []
    num_processed_bits_for_packets = 0

    print("\n--- Анализ и Декодирование Пакетов для Intelligent Voting ---")
    print(f"{'Pkt #':<6} | {'Type':<7} | {'Status':<30} | {'Corrected':<10} | {'Payload HEX':<18}")
    print("-" * (6 + 8 + 31 + 11 + 19))

    for i in range(max_expected_packets):
        is_first_packet = (i == 0)
        use_ecc_for_this_packet = is_first_packet and effective_expect_hybrid_ecc and ecc_possible_for_first and ue

        packet_type_for_info = "ECC" if use_ecc_for_this_packet else "RAW"
        len_bits_to_take = actual_codeword_len_if_ecc if use_ecc_for_this_packet else packet_len_if_raw

        if num_processed_bits_for_packets + len_bits_to_take > len(valid_bits_for_decoding):
            logging.warning(
                f"Недостаточно бит для пакета {i + 1} (ожидалось {len_bits_to_take}, доступно {len(valid_bits_for_decoding) - num_processed_bits_for_packets}).")
            break

        current_input_bits_for_packet = valid_bits_for_decoding[
                                        num_processed_bits_for_packets: num_processed_bits_for_packets + len_bits_to_take]
        num_processed_bits_for_packets += len_bits_to_take

        packet_details_entry: Dict[str, Any] = {
            'payload_bits': None,
            'packet_type': packet_type_for_info,
            'corrected_errors': 0,  # По умолчанию для RAW или если ECC не сработал/не применялся
            'source_packet_index': i,  # Добавим индекс для отладки
            'hex': "N/A"  # Для лога
        }
        status_msg_for_log = "Failed to process"

        if use_ecc_for_this_packet:
            # decode_ecc возвращает (Optional[List[int] сообщения], int исправленных_ошибок)
            message_bits, corrected_count = decode_ecc(current_input_bits_for_packet, bch_code, plb)
            packet_details_entry['corrected_errors'] = corrected_count

            if message_bits is not None and len(message_bits) == payload_len_bits:
                packet_details_entry['payload_bits'] = message_bits
                # Конвертируем в байты для HEX только если биты валидны
                temp_bytes = bits_to_bytes_strict(message_bits, plb)
                if temp_bytes: packet_details_entry['hex'] = temp_bytes.hex()
                status_msg_for_log = f"OK (ECC, {corrected_count if corrected_count != -1 else 'N/A'} fixed)"
            else:
                if message_bits is None:  # Ошибка декодирования ECC
                    status_msg_for_log = f"Fail (ECC Uncorrectable)" if corrected_count == -1 else f"Fail (ECC Decode Err, c:{corrected_count})"
                else:  # Длина полезной нагрузки после декодирования не совпала
                    status_msg_for_log = f"Fail (ECC Decoded Msg Len Mismatch: {len(message_bits)}!={payload_len_bits})"
                    logging.error(f"Pkt {i + 1} (ECC): ошибка длины сообщения {len(message_bits)}!={payload_len_bits}")
                # 'payload_bits' останется None
        else:  # RAW packet
            if len(current_input_bits_for_packet) >= payload_len_bits:
                raw_msg_bits = current_input_bits_for_packet[:payload_len_bits]
                packet_details_entry['payload_bits'] = raw_msg_bits
                # corrected_errors для RAW остается 0
                temp_bytes = bits_to_bytes_strict(raw_msg_bits, plb)
                if temp_bytes: packet_details_entry['hex'] = temp_bytes.hex()
                status_msg_for_log = "OK (RAW)"
            else:
                status_msg_for_log = f"Fail (RAW too short: got {len(current_input_bits_for_packet)}, need {payload_len_bits})"
                # 'payload_bits' останется None

        # Добавляем в общий список только если payload_bits не None (т.е. пакет успешно обработан до уровня сообщения)
        if packet_details_entry['payload_bits'] is not None:
            all_packets_details.append(packet_details_entry)

        # Логирование для каждого пакета
        corrected_display_str = str(packet_details_entry['corrected_errors']) \
            if packet_details_entry['packet_type'] == 'ECC' and packet_details_entry['corrected_errors'] != -1 else "-"
        print(
            f"{i + 1:<6} | {packet_details_entry['packet_type']:<7} | {status_msg_for_log:<30} | {corrected_display_str:<10} | {packet_details_entry['hex']:<18}")

    print("-" * (6 + 8 + 31 + 11 + 19))
    logging.info(f"Intelligent Voting Prep: Собрано {len(all_packets_details)} валидных пакетов для голосования.")

    if not all_packets_details:  # Если после всех попыток нет ни одного валидного пакета
        logging.error("Нет валидных декодированных пакетов для intelligent_voting_strategy.")
        return []

        # --- Вызов Новой Функции Голосования ---
    final_hex_candidates_list = intelligent_voting_strategy(
        valid_packets_info=all_packets_details,
        payload_len_bytes=plb,
        ngram_context_weight=ngram_weight_param  # Используем параметр функции
    )

    end_time = time.time()
    processing_duration_val_run = end_time - start_time
    logging.info(f"Извлечение ЦВЗ завершено. Общее время: {processing_duration_val_run:.2f} сек.")

    if not final_hex_candidates_list:
        logging.error("Intelligent voting не вернуло кандидатов.")
    elif len(final_hex_candidates_list) == 1:
        logging.info(f"Финальный извлеченный ID (1 кандидат): {final_hex_candidates_list[0]}")
    else:
        logging.warning(f"Извлечено несколько возможных кандидатов ID: {', '.join(final_hex_candidates_list)}")

    return final_hex_candidates_list

# --- Функция main ---
# --- Ваша функция main ---
# @profile # Если используете line_profiler
def main() -> int:
    """
    Основная функция запуска экстрактора ЦВЗ.
    Читает необходимое количество кадров с помощью OpenCV и передает их
    функции извлечения, которая использует ThreadPoolExecutor.
    """
    main_start_time = time.time()
    # Используем глобальную LOG_FILENAME
    logging.info(f"--- Запуск Основного Процесса Извлечения (OpenCV Limited Read + ThreadPool) ---")

    # Глобальные флаги доступности библиотек (PYTORCH_WAVELETS_AVAILABLE и т.д.)
    # и объекты (BCH_CODE_OBJECT, device, dtcwt_fwd) должны быть уже инициализированы
    # в блоке if __name__ == "__main__" или глобально в начале скрипта.

    # Проверки основных библиотек (как в вашем коде)
    if not PYTORCH_WAVELETS_AVAILABLE or not TORCH_DCT_AVAILABLE:
        print("ERROR: PyTorch libraries required.");
        logging.critical("Критические PyTorch библиотеки не найдены.")
        return 1

    # Глобальная current_expect_hybrid_ecc или USE_ECC
    # В вашем коде main current_expect_hybrid_ecc берется из globals().get('expect_hybrid_ecc', True)
    # Я оставлю это, но лучше определить expect_hybrid_ecc_global в начале файла.
    current_expect_hybrid_ecc = globals().get('expect_hybrid_ecc_global', USE_ECC)

    if USE_ECC and current_expect_hybrid_ecc and not GALOIS_AVAILABLE:
        # print("\nWARNING: ECC requested for hybrid mode but galois unavailable.") # Уже в __main__
        logging.warning("Galois недоступен, ECC для гибридного режима не будет работать (проверка в main).")

    # Инициализация device и dtcwt_fwd (как в вашем коде main)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device=device)
            logging.info(f"Используется CUDA: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e_cuda_init_main:
            logging.error(f"Ошибка CUDA в main: {e_cuda_init_main}. Переключение на CPU.")
            device = torch.device("cpu")
    else:
        logging.info("Используется CPU.")

    dtcwt_fwd: Optional[DTCWTForward] = None
    DTCWTForwardTypeInMain = globals().get('DTCWTForward')
    if PYTORCH_WAVELETS_AVAILABLE and DTCWTForwardTypeInMain:
        try:
            if DTCWTForwardTypeInMain.__name__ != 'Any':
                dtcwt_fwd = DTCWTForwardTypeInMain(J=1, biort='near_sym_a', qshift='qshift_a').to(device)
                logging.info("PyTorch DTCWTForward instance created in main.")
            else:  # Заглушка
                # dtcwt_fwd = DTCWTForwardTypeInMain() # Не сработает, если Any - это просто type
                logging.warning(
                    "DTCWTForward is Any in main, dtcwt_fwd may remain None if not properly initialized globally.")
                # Если dtcwt_fwd останется None, extract_watermark_from_video должна это обработать
        except Exception as e_dtcwt_in_main:
            logging.critical(f"Failed to init DTCWTForward in main: {e_dtcwt_in_main}", exc_info=True)
            return 1
    elif PYTORCH_WAVELETS_AVAILABLE and not DTCWTForwardTypeInMain:
        logging.critical("Тип DTCWTForward не найден (None) в main, хотя PYTORCH_WAVELETS_AVAILABLE=True.")
        return 1
    else:
        logging.critical("PyTorch Wavelets недоступен (проверено в main)!")
        return 1

    # BCH_CODE_OBJECT используется extract_watermark_from_video,
    # он должен быть инициализирован в __main__ и быть доступен глобально.
    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is None:
        logging.warning("BCH_CODE_OBJECT is None в main. ECC может не работать.")

    # Определение имени входного файла (используем глобальные константы)
    input_base = f"watermarked_ffmpeg_t{BCH_T}"  # Глобальная BCH_T
    input_video_path = input_base + INPUT_EXTENSION  # Глобальная INPUT_EXTENSION

    # Загрузка оригинального ID из файла (ВАШ КОД)
    original_id: Optional[str] = None
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE, "r", encoding='utf-8') as f_id:
                original_id = f_id.read().strip()
            if not (original_id and len(original_id) == PAYLOAD_LEN_BYTES * 2):
                logging.warning(f"ID в '{ORIGINAL_WATERMARK_FILE}' неверной длины или пуст.")
                original_id = None
            else:
                bytes.fromhex(original_id)  # Проверка на HEX
                logging.info(f"Original ID loaded из TXT: {original_id}")
        except ValueError:
            logging.warning(f"ID в '{ORIGINAL_WATERMARK_FILE}' не является валидным HEX.")
            original_id = None
        except Exception as e_read_id_main:
            logging.error(f"Ошибка чтения ID из TXT: {e_read_id_main}")
            original_id = None
    else:
        logging.warning(f"Файл ID '{ORIGINAL_WATERMARK_FILE}' не найден.")

    logging.info(f"--- Начало извлечения из файла: '{input_video_path}' ---")
    if not os.path.exists(input_video_path):
        logging.critical(f"Входной файл не найден: '{input_video_path}'.")
        print(f"ОШИБКА: Файл не найден: '{input_video_path}'")
        return 1

    # --- Расчет необходимого числа кадров для чтения (ВАШ КОД) ---
    payload_len_bits_calc = PAYLOAD_LEN_BYTES * 8
    packet_len_if_ecc_calc = payload_len_bits_calc
    packet_len_if_raw_calc = payload_len_bits_calc
    ecc_possible_for_first_calc_main = False
    actual_codeword_len_calc_main_run = payload_len_bits_calc  # Инициализация

    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is not None:
        try:
            if hasattr(BCH_CODE_OBJECT, 'k') and hasattr(BCH_CODE_OBJECT, 'n'):
                if payload_len_bits_calc <= BCH_CODE_OBJECT.k:
                    actual_codeword_len_calc_main_run = BCH_CODE_OBJECT.n
                    ecc_possible_for_first_calc_main = True
        except Exception:
            pass

    expect_hybrid_for_calc_run = current_expect_hybrid_ecc
    if expect_hybrid_for_calc_run and not ecc_possible_for_first_calc_main:
        expect_hybrid_for_calc_run = False
    max_possible_bits_run = 0
    if expect_hybrid_for_calc_run:
        max_possible_bits_run = actual_codeword_len_calc_main_run + max(0,
                                                                        MAX_TOTAL_PACKETS_global - 1) * packet_len_if_raw_calc
    else:
        current_packet_len_for_calc_run = actual_codeword_len_calc_main_run if USE_ECC and ecc_possible_for_first_calc_main and not expect_hybrid_for_calc_run else packet_len_if_raw_calc
        max_possible_bits_run = MAX_TOTAL_PACKETS_global * current_packet_len_for_calc_run
    if BITS_PER_PAIR <= 0: logging.critical("BITS_PER_PAIR <= 0!"); return 1
    pairs_needed_for_extract = math.ceil(max_possible_bits_run / BITS_PER_PAIR) if max_possible_bits_run > 0 else 0
    if pairs_needed_for_extract == 0: logging.error("Расчет: 0 пар."); return 1
    num_frames_to_read = pairs_needed_for_extract * 2
    logging.info(f"Требуется {pairs_needed_for_extract} пар, читаем {num_frames_to_read} кадров.")

    read_start_time = time.time()
    # В вашем коде был вызов read_required_frames_pyav
    logging.info(f"Чтение первых {num_frames_to_read} кадров с помощью PyAV из '{input_video_path}'...")
    video_stream_idx_to_read: Optional[int] = 0
    frames_for_extraction = read_required_frames_pyav(input_video_path, num_frames_to_read,
                                                      preferred_video_stream_index=video_stream_idx_to_read)
    read_time = time.time() - read_start_time

    if frames_for_extraction is None: logging.critical("Ошибка чтения кадров."); return 1
    logging.info(f"Прочитано {len(frames_for_extraction)} кадров за {read_time:.2f} сек.")
    if len(frames_for_extraction) < 2: logging.error(f"Слишком мало кадров: {len(frames_for_extraction)}."); return 1

    # --- Вызов extract_watermark_from_video (ПРЕДПОЛАГАЕТСЯ, ЧТО ОНА ВОЗВРАЩАЕТ Optional[bytes]) ---
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
        # ngram_weight_param не передаем, так как старая extract_watermark_from_video его не ждет
    )
    # ------------------------------------------------------------------------------------------
    if frames_for_extraction: del frames_for_extraction; gc.collect()

    # --- Вывод результатов (ВАШ КОД, КОТОРЫЙ ОЖИДАЕТ Optional[bytes]) ---
    print(f"\n--- Extraction Results ---")
    extracted_hex: Optional[str] = None
    if extracted_bytes:  # extracted_bytes это Optional[bytes]
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
    # original_id был загружен в начале функции main
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
    # ----------------------------------------------------

    logging.info("--- Extraction Main Process Finished ---")
    total_main_time = time.time() - main_start_time
    logging.info(f"--- Total Extractor Time: {total_main_time:.2f} sec ---")
    print(f"\nExtraction finished. Log: {LOG_FILENAME}")  # Глобальная LOG_FILENAME
    return 0 if final_match_status else 1  # ИЗМЕНЕНО: 0 при успехе, 1 при ошибке/несовпадении


# --- Блок if __name__ == "__main__": (ВАШ КОД С МИНИМАЛЬНЫМИ ИЗМЕНЕНИЯМИ ДЛЯ ИНИЦИАЛИЗАЦИИ) ---
if __name__ == "__main__":
    # --- Настройка логирования (ВАШ КОД) ---
    # LOG_FILENAME должна быть определена глобально
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(filename=LOG_FILENAME,
                            filemode='w',
                            level=logging.INFO,
                            format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # Устанавливаем уровень здесь

    # --- Проверка зависимостей (ВАШ КОД, но с CV2_AVAILABLE) ---
    missing_libs_critical = []
    # Флаги PYAV_AVAILABLE, PYTORCH_WAVELETS_AVAILABLE, TORCH_DCT_AVAILABLE, CV2_AVAILABLE, GALOIS_IMPORTED
    # должны быть установлены глобально в начале файла при попытках импорта.
    if not globals().get('PYAV_AVAILABLE', False): missing_libs_critical.append("PyAV (av)")
    if not globals().get('PYTORCH_WAVELETS_AVAILABLE', False): missing_libs_critical.append("pytorch_wavelets")
    if not globals().get('TORCH_DCT_AVAILABLE', False): missing_libs_critical.append("torch-dct")
    if not globals().get('CV2_AVAILABLE', False): missing_libs_critical.append("OpenCV (cv2)")  # Проверяем флаг
    try:
        import numpy
    except ImportError:
        missing_libs_critical.append("NumPy")
    try:
        import torch
    except ImportError:
        missing_libs_critical.append("PyTorch")
    # Остальные стандартные импорты (shutil, subprocess, hashlib и т.д.) обычно всегда доступны

    if missing_libs_critical:
        error_msg = f"ОШИБКА: Отсутствуют КРИТИЧЕСКИ важные библиотеки: {', '.join(missing_libs_critical)}."
        print(error_msg);
        logging.critical(error_msg);
        sys.exit(1)

    # Предупреждение о Galois (ВАШ КОД)
    if globals().get('USE_ECC', True) and not globals().get('GALOIS_AVAILABLE', False):  # GALOIS_AVAILABLE тоже флаг
        print("\nПРЕДУПРЕЖДЕНИЕ: USE_ECC=True, но библиотека 'galois' не найдена/не работает.")
        logging.warning("ECC включен, но Galois недоступен (проверено в __main__).")

    # --- Инициализация BCH_CODE_OBJECT (ВАШ КОД) ---
    # BCH_M и BCH_T должны быть определены глобально
    if USE_ECC and GALOIS_AVAILABLE and BCH_CODE_OBJECT is None:  # type: ignore
        try:
            _m_run = BCH_M  # type: ignore
            _t_run = BCH_T  # type: ignore
            _n_bch_run = (1 << _m_run) - 1
            _d_bch_run = 2 * _t_run + 1
            expected_k_run = -1
            if _t_run == 5:
                expected_k_run = 215
            elif _t_run == 7:
                expected_k_run = 201
            elif _t_run == 9:
                expected_k_run = 187
            elif _t_run == 11:
                expected_k_run = 173
            elif _t_run == 15:
                expected_k_run = 131

            if expected_k_run != -1 and GALOIS_IMPORTED:  # type: ignore
                BCHTypeRun = globals().get('BCH_TYPE')
                if BCHTypeRun and BCHTypeRun.__name__ != 'BCHPlaceholder':  # Проверка, что это не заглушка
                    temp_bch_run_obj = BCHTypeRun(_n_bch_run, d=_d_bch_run)
                    if hasattr(temp_bch_run_obj, 'k') and hasattr(temp_bch_run_obj, 't') and \
                            temp_bch_run_obj.k == expected_k_run and temp_bch_run_obj.t == _t_run:
                        BCH_CODE_OBJECT = temp_bch_run_obj  # Присваиваем глобальной переменной
                        logging.info(
                            f"BCH_CODE_OBJECT инициализирован в __main__: n={BCH_CODE_OBJECT.n}, k={BCH_CODE_OBJECT.k}, t={BCH_CODE_OBJECT.t}")  # type: ignore
                    else:
                        logging.error(f"Ошибка инициализации BCH в __main__: параметры k/t не совпали.")
            else:
                logging.error(
                    f"Не удалось инициализировать BCH_CODE_OBJECT в __main__ (expected_k или GALOIS_IMPORTED).")
        except Exception as e_bch_init_main_block:
            logging.error(f"Ошибка при инициализации BCH_CODE_OBJECT в __main__: {e_bch_init_main_block}",
                          exc_info=True)
            BCH_CODE_OBJECT = None

            # --- Профилирование (ВАШ КОД) ---
    DO_PROFILING = False
    profiler_instance = None
    if DO_PROFILING:
        if 'KERNPROF_VAR' not in os.environ and (
                'profile' not in globals() or not callable(globals().get('profile'))) and 'cProfile' in sys.modules:
            try:
                import cProfile
                import pstats

                profiler_instance = cProfile.Profile()
                profiler_instance.enable()
                print("cProfile профилирование включено.")
            except ImportError:
                DO_PROFILING = False  # Не удалось импортировать

    final_exit_code = 1
    try:
        final_exit_code = main()
    except FileNotFoundError as e_fnf_main_run:
        print(f"\nОШИБКА: Файл не найден: {e_fnf_main_run}")
        logging.critical(f"FileNotFoundError в __main__: {e_fnf_main_run}", exc_info=True)
    except torch.cuda.OutOfMemoryError as e_oom_main_run:  # type: ignore
        print(f"\nОШИБКА: Недостаточно памяти CUDA: {e_oom_main_run}")
        logging.critical(f"torch.cuda.OutOfMemoryError в __main__: {e_oom_main_run}", exc_info=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()  # type: ignore
    except Exception as e_global_main_run:
        print(f"\nКРИТИЧЕСКАЯ НЕОБРАБОТАННАЯ ОШИБКА: {e_global_main_run}")
        logging.critical(f"Необработанная ошибка в __main__: {e_global_main_run}", exc_info=True)
    finally:
        if DO_PROFILING and profiler_instance is not None:
            profiler_instance.disable()
            logging.info("cProfile профилирование выключено.")
            try:
                import pstats

                stats_obj_final_run = pstats.Stats(profiler_instance).strip_dirs().sort_stats("cumulative")
                print("\n--- Статистика Профилирования (cProfile, Top 30) ---");
                stats_obj_final_run.print_stats(30)
                # Ваш код сохранения статистики
                bch_t_prof_val = globals().get('BCH_T', 'X')
                profile_prof_file_val = f"profile_extract_main_t{bch_t_prof_val}.prof"
                profile_txt_file_val = f"profile_extract_main_t{bch_t_prof_val}.txt"
                stats_obj_final_run.dump_stats(profile_prof_file_val)
                with open(profile_txt_file_val, 'w', encoding='utf-8') as f_pstats_val:
                    ps_val = pstats.Stats(profiler_instance, stream=f_pstats_val).strip_dirs().sort_stats('cumulative')
                    ps_val.print_stats()
                print(f"Статистика профилирования сохранена: {profile_prof_file_val}, {profile_txt_file_val}")
            except Exception as e_pstats_save_val:
                logging.error(f"Ошибка сохранения статистики профилирования: {e_pstats_save_val}")

        logging.info(f"Скрипт watermark_extractor.py завершен с кодом выхода {final_exit_code}.")
        print(f"\nСкрипт завершен с кодом выхода {final_exit_code}.")
        sys.exit(final_exit_code)
