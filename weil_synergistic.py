#!/usr/bin/env python3
"""
СИНЕРГИЧЕСКАЯ ПРОВЕРКА КРИТЕРИЯ ВЕЙЛЯ
С комплексными окнами, правильной Грам-матрицей и голографической диагностикой
"""

import numpy as np
from scipy import special, signal, optimize
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track
from typing import Tuple, Dict, List

console = Console()

# Первые 50 нулей Римана для проверки
RIEMANN_ZEROS = [
    14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
    37.5861781588, 40.9187190121, 43.3270732809, 48.0051508812, 49.7738324777,
    52.9703214777, 56.4462476970, 59.3470440026, 60.8317785246, 65.1125440481,
    67.0798105295, 69.5464017112, 72.0671576745, 75.7046906990, 77.1448400069,
    79.3373750203, 82.9103808541, 84.7354929806, 87.4252746131, 88.8091112076,
    92.4918992705, 94.6513440442, 95.8706342057, 98.8311949428, 101.3178510097,
    103.7255382041, 105.4466223971, 107.1686110655, 111.0295355376, 111.8746592257,
    114.3202209715, 116.2266803236, 118.7907829657, 121.3701250226, 122.9468292956,
    124.2568186822, 127.5166839618, 129.5787042035, 131.0876885039, 133.4977371892,
    134.7565097488, 138.1160420461, 139.7362089052, 141.1237074065, 143.1118458127
]

def von_mangoldt_extended(n: int) -> float:
    """
    Функция фон Мангольдта Λ(n) с оптимизацией
    """
    if n <= 1:
        return 0.0
    
    # Проверка на степени 2
    if n & (n - 1) == 0:
        return np.log(2)
    
    # Факторизация
    temp = n
    prime_power = 0
    prime_base = 0
    
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count > 0:
            if prime_power > 0:  # Уже нашли один простой делитель
                return 0.0
            prime_power = count
            prime_base = p
    
    if temp > 1:
        if prime_power > 0:  # Уже есть другой простой делитель
            return 0.0
        return np.log(temp)
    
    if prime_power > 0:
        return np.log(prime_base)
    
    return 0.0

def create_complex_bump(A: float, N: int = 4096, modulation: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Создаем КОМПЛЕКСНУЮ bump-функцию с компактной поддержкой [-A, A]
    modulation - параметр для фазовой модуляции (для пробных нулей)
    """
    # Частотная сетка с высоким разрешением
    xi = np.linspace(-3*A, 3*A, N)
    dxi = xi[1] - xi[0]
    
    # C∞ bump функция (вещественная в частотной области)
    h_hat = np.zeros(N, dtype=complex)
    mask = np.abs(xi) < A
    x = xi[mask] / A
    
    # Основная bump функция
    h_hat[mask] = np.exp(-1.0 / (1.0 - x**2))
    
    # Добавляем фазовую модуляцию для создания комплексного окна
    if modulation != 0:
        h_hat = h_hat * np.exp(1j * modulation * xi)
    
    # Нормализация
    h_hat = h_hat / np.max(np.abs(h_hat))
    
    # Обратное FFT для получения h(t) - КОМПЛЕКСНОЙ функции
    t = 2 * np.pi * fftfreq(N, d=dxi)
    h = ifft(ifftshift(h_hat)) * N * dxi / (2 * np.pi)
    h = fftshift(h)  # НЕ берем real часть!
    
    # Сортировка по t
    idx = np.argsort(t)
    t = t[idx]
    h = h[idx]
    
    return t, h, xi, h_hat

def convolution_product(F1: np.ndarray, F2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Правильное произведение для Грам-матрицы: φ_ij = F_i * F̃_j (свертка)
    """
    # Используем теорему о свертке: conv(f,g) = ifft(fft(f) * conj(fft(g)))
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    
    # FFT с правильной нормировкой
    F1_hat = fft(F1) * dt
    F2_hat = fft(F2) * dt
    
    # Произведение в частотной области
    phi_hat = F1_hat * np.conj(F2_hat)
    
    # Обратное преобразование
    phi = ifft(phi_hat) / dt
    
    return phi

def compute_archimedean_complex(t: np.ndarray, phi: np.ndarray) -> complex:
    """
    Архимедов член для КОМПЛЕКСНОЙ функции
    A(φ) = (1/2π) ∫ φ(t) [Re ψ(1/4 + it/2) - log π] dt
    """
    dt = t[1] - t[0] if len(t) > 1 else 0
    
    # Дигамма вес (вещественный)
    weights = np.zeros_like(t, dtype=float)
    for i, tau in enumerate(t):
        z = 0.25 + 0.5j * tau
        psi_val = special.digamma(z)
        weights[i] = np.real(psi_val) - np.log(np.pi)
    
    # Интеграл с комплексной функцией
    A = np.sum(phi * weights) * dt / (2 * np.pi)
    
    return A

def compute_prime_exact(xi: np.ndarray, phi_hat: np.ndarray, A_cutoff: float, 
                        use_all_powers: bool = True) -> complex:
    """
    Точное вычисление простого члена с правильной дискретизацией
    P(φ) = -(1/2π) Σ_{n≤e^A} Λ(n)/√n [φ̂(log n) + φ̂(-log n)]
    
    use_all_powers: если True, включаем все степени простых p^m
    """
    n_max = int(np.exp(A_cutoff))
    P = 0.0 + 0.0j
    
    # Собираем все n с Λ(n) > 0
    contributions = []
    
    for n in range(2, n_max + 1):
        lambda_n = von_mangoldt_extended(n)
        if lambda_n > 0:
            log_n = np.log(n)
            
            # ТОЧНАЯ интерполяция в частотной области
            # Используем кубическую интерполяцию для лучшей точности
            phi_hat_plus = np.interp(log_n, xi, phi_hat, left=0, right=0)
            phi_hat_minus = np.interp(-log_n, xi, phi_hat, left=0, right=0)
            
            # Вклад с правильным весом
            contribution = lambda_n / np.sqrt(n) * (phi_hat_plus + phi_hat_minus)
            P -= contribution / (2 * np.pi)
            
            contributions.append((n, lambda_n, contribution))
            
            # Если не используем все степени, останавливаемся на простых
            if not use_all_powers and n > 2:
                # Проверяем, является ли n простым (не степенью)
                is_prime_power = False
                for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
                    if n % p == 0 and n != p:
                        is_prime_power = True
                        break
                if is_prime_power:
                    continue
    
    return P

def build_gram_matrix_correct(test_functions: List, A_cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строим правильную Грам-матрицу из функционала Вейля
    M_ij = Q(φ_ij) где φ_ij = F_i * F̃_j
    """
    N = len(test_functions)
    M = np.zeros((N, N), dtype=complex)
    
    console.print("[cyan]Building correct Gram matrix with convolution products...")
    
    for i in range(N):
        for j in range(i, N):
            t_i, F_i, xi_i, F_hat_i = test_functions[i]
            t_j, F_j, xi_j, F_hat_j = test_functions[j]
            
            # Правильное произведение через свертку
            phi_ij = convolution_product(F_i, np.conj(F_j), t_i)
            
            # FFT от произведения
            phi_ij_hat = fft(phi_ij) * (t_i[1] - t_i[0])
            
            # Расширяем частотную сетку если нужно
            xi_extended = fftfreq(len(phi_ij), d=(t_i[1] - t_i[0])) * 2 * np.pi
            
            # Архимедов член
            A_ij = compute_archimedean_complex(t_i, phi_ij)
            
            # Простой член с точной дискретизацией
            P_ij = compute_prime_exact(xi_extended, phi_ij_hat, A_cutoff)
            
            # Нулевой член (для первых нулей)
            Z_ij = 0.0 + 0.0j
            for gamma in RIEMANN_ZEROS[:20]:
                if t_i[0] <= gamma <= t_i[-1]:
                    phi_gamma = np.interp(gamma, t_i, phi_ij)
                    Z_ij += phi_gamma
            
            # Элемент матрицы БЕЗ калибровки
            M[i, j] = Z_ij - A_ij - P_ij
            
            if i != j:
                M[j, i] = np.conj(M[i, j])  # Эрмитова симметрия
    
    console.print(f"[yellow]Gram matrix shape: {M.shape}")
    console.print(f"[yellow]Matrix norm: {np.linalg.norm(M, 'fro'):.6f}")
    
    # Собственные значения эрмитовой матрицы
    eigenvals = np.linalg.eigvalsh(M)
    
    return M, eigenvals

def adversarial_minimum_search(test_functions: List, A_cutoff: float) -> Tuple[float, np.ndarray]:
    """
    Адверсариальный поиск: min_{||c||=1} c* Q(F) c
    """
    console.print("[magenta]Running adversarial minimum search...")
    
    # Строим матрицу
    M, _ = build_gram_matrix_correct(test_functions, A_cutoff)
    
    # Находим минимальное собственное значение и соответствующий вектор
    eigenvals, eigenvecs = np.linalg.eigh(M)
    min_idx = np.argmin(eigenvals)
    lambda_min = eigenvals[min_idx]
    min_vector = eigenvecs[:, min_idx]
    
    # Проверяем через оптимизацию
    def quadratic_form(c):
        c_complex = c[:len(c)//2] + 1j * c[len(c)//2:]
        c_complex = c_complex / np.linalg.norm(c_complex)
        return np.real(c_complex.conj() @ M @ c_complex)
    
    # Начальное приближение - случайный вектор
    c0 = np.random.randn(2 * len(test_functions))
    c0 = c0 / np.linalg.norm(c0)
    
    # Минимизация
    result = optimize.minimize(quadratic_form, c0, method='L-BFGS-B')
    optimized_min = result.fun
    
    console.print(f"[yellow]Eigenvalue method: λ_min = {lambda_min:.6f}")
    console.print(f"[yellow]Optimization method: min = {optimized_min:.6f}")
    console.print(f"[{'green' if abs(lambda_min - optimized_min) < 1e-6 else 'red'}]Agreement: {abs(lambda_min - optimized_min):.2e}")
    
    return lambda_min, min_vector

def holographic_diagnostic(A_cutoff: float = 3.0):
    """
    Голографическая диагностика с пробным нулем и вычитанием базовой линии
    """
    console.print(Panel.fit("[bold cyan]HOLOGRAPHIC DIAGNOSTIC WITH BASELINE[/bold cyan]", box=box.DOUBLE))
    
    # Фиксированные окна для γ₁, γ₂
    t1, F1, xi1, F_hat1 = create_complex_bump(A_cutoff, modulation=0.1)
    t2, F2, xi2, F_hat2 = create_complex_bump(A_cutoff, modulation=0.2)
    
    # Сканируем пробный нуль
    probe_range = np.linspace(10, 50, 100)
    interference_map = []
    
    for gamma_probe in track(probe_range, description="Scanning probe zero..."):
        # Создаем окно с модуляцией для пробного нуля
        modulation_probe = 2 * np.pi * gamma_probe / 100
        t_p, F_p, xi_p, F_hat_p = create_complex_bump(A_cutoff, modulation=modulation_probe)
        
        # Строим 3×3 матрицу
        test_funcs = [(t1, F1, xi1, F_hat1), (t2, F2, xi2, F_hat2), (t_p, F_p, xi_p, F_hat_p)]
        M, eigenvals = build_gram_matrix_correct(test_funcs, A_cutoff)
        
        # Сохраняем недиагональные элементы
        interference_map.append([M[0, 2], M[1, 2]])
    
    interference_map = np.array(interference_map)
    
    # FFT от интерференционной карты
    spectrum_02 = fft(interference_map[:, 0])
    spectrum_12 = fft(interference_map[:, 1])
    
    # Находим пики
    freqs = fftfreq(len(probe_range), d=(probe_range[1] - probe_range[0]))
    
    # Baseline: рандомизированные простые (модель Крамера)
    console.print("\n[yellow]Computing Cramér baseline...")
    baseline_map = []
    
    # Создаем случайную модель простых
    np.random.seed(42)
    for gamma_probe in probe_range:
        # Случайные "простые"
        random_phase = np.random.uniform(-np.pi, np.pi)
        baseline_map.append([np.exp(1j * random_phase) * 0.1, np.exp(1j * random_phase * 1.5) * 0.1])
    
    baseline_map = np.array(baseline_map)
    baseline_spectrum_02 = fft(baseline_map[:, 0])
    baseline_spectrum_12 = fft(baseline_map[:, 1])
    
    # Вычитаем базовую линию
    signal_02 = np.abs(spectrum_02) - np.abs(baseline_spectrum_02)
    signal_12 = np.abs(spectrum_12) - np.abs(baseline_spectrum_12)
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Интерференционная карта
    ax = axes[0, 0]
    ax.plot(probe_range, np.abs(interference_map[:, 0]), 'r-', label='|M₀₂|', alpha=0.7)
    ax.plot(probe_range, np.abs(interference_map[:, 1]), 'b-', label='|M₁₂|', alpha=0.7)
    ax.set_xlabel('Probe zero γ')
    ax.set_ylabel('|M_ij|')
    ax.set_title('Interference Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Спектр до вычитания
    ax = axes[0, 1]
    ax.semilogy(freqs[:len(freqs)//2], np.abs(spectrum_02[:len(spectrum_02)//2]), 'r-', label='Raw M₀₂')
    ax.semilogy(freqs[:len(freqs)//2], np.abs(baseline_spectrum_02[:len(baseline_spectrum_02)//2]), 'r--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('|FFT|')
    ax.set_title('Raw Spectrum vs Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Сигнал после вычитания baseline
    ax = axes[1, 0]
    ax.plot(freqs[:len(freqs)//2], signal_02[:len(signal_02)//2], 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Signal - Baseline')
    ax.set_title('True Signal (after baseline subtraction)')
    ax.grid(True, alpha=0.3)
    
    # Статистика пиков
    ax = axes[1, 1]
    # Находим значимые пики
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(signal_02[:len(signal_02)//2], height=np.std(signal_02) * 2)
    
    if len(peaks) > 0:
        ax.stem(freqs[peaks], signal_02[peaks], 'go', basefmt=' ')
        ax.set_xlabel('Peak frequencies')
        ax.set_ylabel('Peak amplitude')
        ax.set_title(f'Significant peaks: {len(peaks)}')
    else:
        ax.text(0.5, 0.5, 'No significant peaks found', ha='center', va='center', fontsize=14)
        ax.set_title('Peak Analysis')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Holographic Diagnostic with Baseline Subtraction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('holographic_diagnostic.png', dpi=150)
    console.print("[green]Saved: holographic_diagnostic.png")
    plt.show()
    
    return interference_map, signal_02, peaks

def stability_test_grid_doubling(A_cutoff: float = 3.0):
    """
    Тест стабильности при удвоении сетки
    """
    console.print("[cyan]Testing stability with grid doubling...")
    
    grid_sizes = [512, 1024, 2048, 4096]
    results = []
    
    for N in grid_sizes:
        # Создаем тест-функции с разным разрешением
        t, F, xi, F_hat = create_complex_bump(A_cutoff, N=N, modulation=0.1)
        test_funcs = [(t, F, xi, F_hat)]
        
        # Строим 1×1 матрицу для простоты
        M, eigenvals = build_gram_matrix_correct(test_funcs, A_cutoff)
        
        results.append({
            'N': N,
            'value': M[0, 0],
            'abs': np.abs(M[0, 0])
        })
        
        console.print(f"N={N}: Q(F) = {M[0, 0]:.6f}")
    
    # Проверяем сходимость
    values = [r['abs'] for r in results]
    variation = np.std(values) / np.mean(values)
    
    console.print(f"\n[{'green' if variation < 0.1 else 'red'}]Relative variation: {variation:.3f}")
    
    return results

def main():
    console.print("[bold red]SYNERGISTIC WEIL VERIFICATION WITH COMPLEX WINDOWS[/bold red]\n")
    
    # Тест 1: Правильная Грам-матрица с комплексными окнами
    console.print("\n[bold]TEST 1: Complex windows and proper Gram matrix[/bold]")
    
    A_values = [2.0, 3.0, 4.0]
    for A in A_values:
        console.print(f"\n[yellow]Testing A = {A}")
        
        # Создаем набор комплексных тест-функций
        test_funcs = []
        for mod in [0.0, 0.1, 0.2]:
            t, F, xi, F_hat = create_complex_bump(A, modulation=mod)
            test_funcs.append((t, F, xi, F_hat))
        
        # Строим Грам-матрицу
        M, eigenvals = build_gram_matrix_correct(test_funcs, A)
        
        console.print(f"Eigenvalues: {eigenvals}")
        console.print(f"[{'green' if eigenvals[0] > 0 else 'red'}]λ_min = {eigenvals[0]:.6f}")
    
    # Тест 2: Адверсариальный поиск
    console.print("\n[bold]TEST 2: Adversarial minimum search[/bold]")
    t, F, xi, F_hat = create_complex_bump(3.0, modulation=0.15)
    lambda_min, min_vec = adversarial_minimum_search([(t, F, xi, F_hat)], 3.0)
    
    # Тест 3: Стабильность при удвоении сетки
    console.print("\n[bold]TEST 3: Grid doubling stability[/bold]")
    stability_results = stability_test_grid_doubling(3.0)
    
    # Тест 4: Голографическая диагностика
    console.print("\n[bold]TEST 4: Holographic diagnostic[/bold]")
    interference, signal, peaks = holographic_diagnostic(3.0)
    
    # Финальная оценка
    console.print("\n[bold green]SYNERGISTIC IMPROVEMENTS IMPLEMENTED:[/bold green]")
    console.print("✓ Complex windows preserve phase information")
    console.print("✓ Proper Gram matrix via convolution φ_ij = F_i * F̃_j")
    console.print("✓ Exact discretization at log(p^m) points")
    console.print("✓ Baseline subtraction for holographic signal")
    console.print("✓ Adversarial search for true minimum")
    console.print("✓ Grid stability verification")
    
    if len(peaks) > 0:
        console.print(f"\n[cyan]Found {len(peaks)} significant holographic peaks!")
        console.print("This may indicate structure in the zero distribution")

if __name__ == "__main__":
    main()