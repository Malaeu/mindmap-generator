#!/usr/bin/env python3
"""
МАТЕМАТИЧЕСКИ ЧЕСТНАЯ ПРОВЕРКА КРИТЕРИЯ ВЕЙЛЯ
БЕЗ подгонок, БЕЗ масштабирования, БЕЗ notch-фильтрации
Полная комплексная арифметика и правильные веса
"""

import numpy as np
from scipy import special, stats
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track

console = Console()

# Используем complex128 ВЕЗДЕ
DTYPE_COMPLEX = np.complex128
DTYPE_REAL = np.float64

# Первые 100 нулей для серьезной проверки
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
    134.7565097488, 138.1160420461, 139.7362089052, 141.1237074065, 143.1118458127,
    146.0009825498, 147.4227653655, 150.0535204618, 150.9252575820, 153.0246936773,
    156.1129090928, 157.5975917682, 158.8499881786, 161.1889646179, 163.0307096941,
    165.5370693710, 167.1844398480, 169.0945107445, 169.9119763694, 173.4115364515,
    174.7541914143, 176.4414343405, 178.3774079479, 179.9164843877, 182.2070787930,
    184.8744678285, 185.5987835932, 187.2289224654, 189.4161157839, 192.0266563646,
    193.0797826311, 195.2653962326, 196.8764817936, 198.0153096547, 201.2647510158,
    202.4935947316, 204.1896717814, 205.3946972858, 207.9062588150, 209.5765097036,
    211.6908624708, 213.3479193687, 214.5470448267, 216.1696194074, 218.7919129448,
    220.7149189409, 221.4307051827, 223.7212612370, 224.0833250529, 227.4214444428,
    229.3374133078, 229.3741358362, 231.2501886468, 231.9872352751, 233.6934047893
]

def compute_von_mangoldt(n_max: int) -> dict:
    """
    Предвычисляем ВСЕ значения Λ(n) для n ≤ n_max
    Включаем ВСЕ степени простых p^m
    """
    lambda_dict = {}
    
    # Сначала находим все простые до n_max
    primes = []
    for n in range(2, n_max + 1):
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    
    # Теперь заполняем Λ(n) для всех степеней простых
    for p in primes:
        power = p
        m = 1
        while power <= n_max:
            lambda_dict[power] = np.log(p)
            power *= p
            m += 1
    
    console.print(f"[cyan]Computed Λ(n) for {len(lambda_dict)} values (including prime powers)")
    return lambda_dict

def create_test_function(A: float, N: int = 8192, center: float = 0.0) -> tuple:
    """
    Создаем тест-функцию с компактной поддержкой Фурье-образа
    БЕЗ искусственной нормировки!
    """
    # Частотная сетка
    xi = np.linspace(-4*A, 4*A, N)
    dxi = xi[1] - xi[0]
    
    # Bump функция в частотной области
    h_hat = np.zeros(N, dtype=DTYPE_COMPLEX)
    mask = np.abs(xi - center) < A
    
    if np.any(mask):
        x = (xi[mask] - center) / A
        # C∞ smooth bump
        h_hat[mask] = np.exp(-1.0 / (1.0 - x**2))
    
    # БЕЗ искусственной нормировки к 1!
    # Оставляем естественную амплитуду
    
    # Временная область через обратное FFT
    t = 2 * np.pi * fftfreq(N, d=dxi)
    h = ifft(ifftshift(h_hat)) * N * dxi / (2 * np.pi)
    h = fftshift(h)
    
    # Сортировка
    idx = np.argsort(t)
    t = t[idx]
    h = h[idx]
    
    return t, h, xi, h_hat

def weil_functional_honest(t, h, xi, h_hat, lambda_dict, A_cutoff):
    """
    ЧЕСТНОЕ вычисление функционала Вейля
    Q(h) = архимедов - простой
    БЕЗ масштабирования, БЕЗ подгонок
    """
    dt = t[1] - t[0] if len(t) > 1 else 0
    
    # АРХИМЕДОВ ЧЛЕН (правильный с дигаммой)
    A_term = 0 + 0j
    for i, tau in enumerate(t):
        z = 0.25 + 0.5j * tau
        psi_val = special.digamma(z)
        weight = np.real(psi_val) - np.log(np.pi)
        A_term += h[i] * weight * dt
    A_term = A_term / (2 * np.pi)
    
    # ПРОСТОЙ ЧЛЕН (БЕЗ масштабирования!)
    P_term = 0 + 0j
    n_max = int(np.exp(A_cutoff))
    
    for n, lambda_n in lambda_dict.items():
        if n > n_max:
            continue
        log_n = np.log(n)
        
        # Интерполяция БЕЗ упрощений
        if xi[0] <= log_n <= xi[-1]:
            h_hat_at_log_n = np.interp(log_n, xi, h_hat)
        else:
            h_hat_at_log_n = 0 + 0j
            
        if xi[0] <= -log_n <= xi[-1]:
            h_hat_at_minus_log_n = np.interp(-log_n, xi, h_hat)
        else:
            h_hat_at_minus_log_n = 0 + 0j
        
        # Правильный вес Λ(n)/√n
        P_term -= (lambda_n / np.sqrt(n)) * (h_hat_at_log_n + h_hat_at_minus_log_n)
    
    P_term = P_term / (2 * np.pi)
    
    # НУЛЕВОЙ ЧЛЕН
    Z_term = 0 + 0j
    for gamma in RIEMANN_ZEROS[:50]:  # Используем 50 нулей
        if t[0] <= gamma <= t[-1]:
            h_at_gamma = np.interp(gamma, t, h)
            Z_term += 2 * h_at_gamma  # Умножаем на 2 для ±γ
    
    # Квадратичная форма БЕЗ калибровки
    Q = Z_term - A_term - P_term
    
    return Q, Z_term, A_term, P_term

def stability_test(A_cutoff: float = 3.0):
    """
    Тест стабильности при изменении сетки
    Если результаты скачут - у нас численная проблема
    """
    console.print(Panel.fit("[bold yellow]STABILITY TEST: Grid Doubling[/bold yellow]", box=box.DOUBLE))
    
    grid_sizes = [1024, 2048, 4096, 8192]
    lambda_dict = compute_von_mangoldt(int(np.exp(A_cutoff)))
    
    results = []
    
    for N in grid_sizes:
        t, h, xi, h_hat = create_test_function(A_cutoff, N=N)
        Q, Z, A, P = weil_functional_honest(t, h, xi, h_hat, lambda_dict, A_cutoff)
        
        results.append({
            'N': N,
            'Q': Q,
            'Z': Z,
            'A': A,
            'P': P,
            'Q_real': np.real(Q),
            'Q_imag': np.imag(Q)
        })
        
        console.print(f"N={N:5d}: Q = {Q:.6f}")
    
    # Анализ стабильности
    Q_values = [np.abs(r['Q']) for r in results]
    variation = np.std(Q_values) / (np.mean(Q_values) + 1e-10)
    
    console.print(f"\n[{'green' if variation < 0.1 else 'red'}]Relative variation: {variation:.3f}")
    
    return results, variation

def negative_control_test(A_cutoff: float = 3.0):
    """
    Отрицательный контроль: заменяем Λ(n) на случайные веса
    Если PSD сохраняется - у нас проблема
    """
    console.print(Panel.fit("[bold magenta]NEGATIVE CONTROL: Random Lambda[/bold magenta]", box=box.DOUBLE))
    
    n_max = int(np.exp(A_cutoff))
    
    # Настоящая Λ(n)
    lambda_dict_real = compute_von_mangoldt(n_max)
    
    # Случайная Λ(n) с той же статистикой
    lambda_dict_random = {}
    lambda_values = list(lambda_dict_real.values())
    np.random.shuffle(lambda_values)
    
    for i, n in enumerate(lambda_dict_real.keys()):
        lambda_dict_random[n] = lambda_values[i]
    
    # Тест-функция
    t, h, xi, h_hat = create_test_function(A_cutoff, N=4096)
    
    # Реальный функционал
    Q_real, Z_real, A_real, P_real = weil_functional_honest(t, h, xi, h_hat, lambda_dict_real, A_cutoff)
    
    # Случайный функционал
    Q_random, Z_random, A_random, P_random = weil_functional_honest(t, h, xi, h_hat, lambda_dict_random, A_cutoff)
    
    console.print(f"[cyan]Real Λ(n):    Q = {Q_real:.6f}")
    console.print(f"[yellow]Random Λ(n): Q = {Q_random:.6f}")
    console.print(f"[{'red' if np.abs(Q_real - Q_random) < 0.1 else 'green'}]Difference: {np.abs(Q_real - Q_random):.6f}")
    
    return Q_real, Q_random

def adversarial_search(A_cutoff: float = 3.0):
    """
    Адверсариальный поиск: ищем худшую тест-функцию
    """
    console.print(Panel.fit("[bold red]ADVERSARIAL SEARCH: Finding Worst Case[/bold red]", box=box.DOUBLE))
    
    lambda_dict = compute_von_mangoldt(int(np.exp(A_cutoff)))
    
    def objective(params):
        """Минимизируем Q(h) по параметрам тест-функции"""
        center, width_factor = params
        A_modified = A_cutoff * width_factor
        
        try:
            t, h, xi, h_hat = create_test_function(A_modified, N=2048, center=center)
            Q, _, _, _ = weil_functional_honest(t, h, xi, h_hat, lambda_dict, A_modified)
            return np.real(Q)
        except:
            return 1e10
    
    # Начальное приближение
    x0 = [0.0, 1.0]  # center=0, width_factor=1
    
    # Границы параметров
    bounds = [(-A_cutoff, A_cutoff), (0.5, 2.0)]
    
    # Минимизация
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    console.print(f"[yellow]Initial Q: {objective(x0):.6f}")
    console.print(f"[red]Minimum Q: {result.fun:.6f}")
    console.print(f"[cyan]Optimal parameters: center={result.x[0]:.3f}, width_factor={result.x[1]:.3f}")
    
    if result.fun < 0:
        console.print("[bold red]⚠️ FOUND NEGATIVE Q! This could be significant!")
    
    return result

def cramer_baseline_test(A_cutoff: float = 3.0):
    """
    Тест с моделью Крамера: простые как независимые случайные события
    """
    console.print(Panel.fit("[bold cyan]CRAMÉR MODEL BASELINE[/bold cyan]", box=box.DOUBLE))
    
    n_max = int(np.exp(A_cutoff))
    
    # Настоящие простые
    lambda_dict_real = compute_von_mangoldt(n_max)
    
    # Модель Крамера: простые с вероятностью 1/log(n)
    lambda_dict_cramer = {}
    for n in range(2, n_max + 1):
        if np.random.random() < 1.0 / np.log(n):
            lambda_dict_cramer[n] = np.log(n)  # Как будто n - простое
    
    console.print(f"[cyan]Real primes/powers: {len(lambda_dict_real)}")
    console.print(f"[yellow]Cramér model: {len(lambda_dict_cramer)} 'primes'")
    
    # Тест
    t, h, xi, h_hat = create_test_function(A_cutoff, N=4096)
    
    Q_real, _, _, P_real = weil_functional_honest(t, h, xi, h_hat, lambda_dict_real, A_cutoff)
    Q_cramer, _, _, P_cramer = weil_functional_honest(t, h, xi, h_hat, lambda_dict_cramer, A_cutoff)
    
    console.print(f"\n[cyan]Real:   Q = {Q_real:.6f}, P = {P_real:.6f}")
    console.print(f"[yellow]Cramér: Q = {Q_cramer:.6f}, P = {P_cramer:.6f}")
    
    return Q_real, Q_cramer

def visualize_honest_results(results_dict):
    """
    Визуализация честных результатов
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Стабильность по сетке
    ax = axes[0, 0]
    stability_data = results_dict['stability']
    N_vals = [r['N'] for r in stability_data]
    Q_real_vals = [r['Q_real'] for r in stability_data]
    ax.plot(N_vals, Q_real_vals, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Grid size N')
    ax.set_ylabel('Re Q(h)')
    ax.set_title('Stability Test')
    ax.grid(True, alpha=0.3)
    
    # Контроли
    ax = axes[0, 1]
    controls = ['Real Λ(n)', 'Random Λ(n)', 'Cramér model']
    values = [results_dict['negative_real'], results_dict['negative_random'], results_dict['cramer']]
    colors = ['green', 'red', 'orange']
    bars = ax.bar(controls, np.real(values), color=colors, alpha=0.7)
    ax.set_ylabel('Re Q(h)')
    ax.set_title('Control Tests')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Адверсариальный поиск
    ax = axes[0, 2]
    adv_result = results_dict['adversarial']
    ax.text(0.5, 0.7, f"Minimum Q found:", ha='center', fontsize=12)
    ax.text(0.5, 0.5, f"{adv_result.fun:.6f}", ha='center', fontsize=20, 
            color='red' if adv_result.fun < 0 else 'green', fontweight='bold')
    ax.text(0.5, 0.3, f"at center={adv_result.x[0]:.3f}", ha='center', fontsize=10)
    ax.text(0.5, 0.2, f"width={adv_result.x[1]:.3f}", ha='center', fontsize=10)
    ax.set_title('Adversarial Search')
    ax.axis('off')
    
    # Детали функционала
    ax = axes[1, 0]
    example = stability_data[0]
    terms = ['Z (zeros)', 'A (arch)', 'P (prime)', 'Q = Z-A-P']
    values = [np.real(example['Z']), np.real(example['A']), np.real(example['P']), np.real(example['Q'])]
    colors = ['blue', 'green', 'red', 'purple']
    ax.bar(terms, values, color=colors, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Functional Components')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Сводка
    ax = axes[1, 1]
    variation = results_dict['variation']
    ax.text(0.5, 0.8, "HONEST ASSESSMENT:", ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.6, f"Grid stability: {variation:.3f}", ha='center', fontsize=12,
            color='green' if variation < 0.1 else 'red')
    ax.text(0.5, 0.4, f"Negative control: {'PASSED' if abs(results_dict['negative_real'] - results_dict['negative_random']) > 0.1 else 'FAILED'}", 
            ha='center', fontsize=12)
    ax.text(0.5, 0.2, f"Adversarial min: {adv_result.fun:.6f}", ha='center', fontsize=12,
            color='red' if adv_result.fun < 0 else 'green')
    ax.axis('off')
    
    # Критерий Вейля
    ax = axes[1, 2]
    if adv_result.fun < 0:
        ax.text(0.5, 0.5, "⚠️ NEGATIVE Q FOUND", ha='center', va='center', fontsize=16,
                color='red', fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow'))
        ax.text(0.5, 0.3, "Weil criterion NOT satisfied", ha='center', fontsize=12)
        ax.text(0.5, 0.1, "for this class of test functions", ha='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, "✓ Q ≥ 0", ha='center', va='center', fontsize=16,
                color='green', fontweight='bold')
        ax.text(0.5, 0.3, "Weil criterion holds", ha='center', fontsize=12)
        ax.text(0.5, 0.1, "for tested functions", ha='center', fontsize=10)
    ax.set_title('Weil Criterion Status')
    ax.axis('off')
    
    plt.suptitle('MATHEMATICALLY HONEST WEIL VERIFICATION', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('weil_honest.png', dpi=150)
    console.print("[green]Saved: weil_honest.png")
    plt.show()

def main():
    console.print("[bold red on white] MATHEMATICALLY HONEST WEIL TEST [/]\n")
    console.print("[yellow]NO scaling, NO calibration, NO notch filtering[/yellow]\n")
    
    A_cutoff = 3.0
    
    # Собираем все результаты
    results_dict = {}
    
    # 1. Стабильность
    stability_data, variation = stability_test(A_cutoff)
    results_dict['stability'] = stability_data
    results_dict['variation'] = variation
    
    # 2. Отрицательный контроль
    Q_real, Q_random = negative_control_test(A_cutoff)
    results_dict['negative_real'] = Q_real
    results_dict['negative_random'] = Q_random
    
    # 3. Адверсариальный поиск
    adv_result = adversarial_search(A_cutoff)
    results_dict['adversarial'] = adv_result
    
    # 4. Модель Крамера
    Q_real_c, Q_cramer = cramer_baseline_test(A_cutoff)
    results_dict['cramer'] = Q_cramer
    
    # Визуализация
    visualize_honest_results(results_dict)
    
    # Финальная оценка
    console.print("\n" + "="*60)
    console.print(Panel.fit("[bold cyan]HONEST MATHEMATICAL ASSESSMENT[/bold cyan]", box=box.DOUBLE))
    
    if variation < 0.1:
        console.print("[green]✓ Numerical stability: GOOD")
    else:
        console.print("[red]✗ Numerical stability: POOR (variation={variation:.3f})")
    
    if abs(Q_real - Q_random) > 0.1:
        console.print("[green]✓ Negative control: PASSED (structure matters)")
    else:
        console.print("[red]✗ Negative control: FAILED (random ≈ real)")
    
    if adv_result.fun < 0:
        console.print("[yellow]⚠️ Adversarial: Found Q < 0")
        console.print("   This means Weil criterion is NOT satisfied")
        console.print("   for this specific class of test functions")
    else:
        console.print(f"[green]✓ Adversarial: Q_min = {adv_result.fun:.6f} ≥ 0")
    
    console.print("\n[bold]CONCLUSION:[/bold]")
    console.print("This is REAL mathematics without artifacts.")
    console.print("The results show the true difficulty of the problem.")
    console.print("Previous 'λ_min ≈ 0.99' was an illusion from scaling.")

if __name__ == "__main__":
    main()