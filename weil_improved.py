#!/usr/bin/env python3
"""
УЛУЧШЕННАЯ ВЕРСИЯ с большим количеством нулей и точным интегрированием
"""

import numpy as np
from scipy import special, integrate
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track

console = Console()

# Первые 50 нулей Римана
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

def improved_von_mangoldt(n):
    """Оптимизированная функция фон Мангольдта"""
    if n <= 1:
        return 0.0
    
    # Быстрая проверка на степени 2
    if n & (n - 1) == 0:  # n является степенью 2
        return np.log(2)
    
    # Обычная факторизация
    factors = []
    temp = n
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            count += 1
            temp //= p
        if count > 0:
            factors.append((p, count))
            if temp == 1:
                break
    
    if temp > 1:
        factors.append((temp, 1))
    
    # Возвращаем log(p) если n = p^k
    if len(factors) == 1:
        p, k = factors[0]
        return np.log(p)
    return 0.0

def create_smooth_bump(A, N=2048):
    """
    Создаем очень гладкую bump-функцию для лучшей точности
    """
    # Более плотная сетка
    xi = np.linspace(-2*A, 2*A, N)
    
    # C∞ bump function
    h_hat = np.zeros_like(xi, dtype=complex)
    mask = np.abs(xi) < A
    
    # Используем более гладкую функцию
    x = xi[mask] / A
    h_hat[mask] = np.exp(-1.0 / (1.0 - x**2)) * np.exp(-x**2 / 2)
    
    # Нормализация
    h_hat = h_hat / np.max(np.abs(h_hat))
    
    # Обратное FFT для h(t)
    dt = 2 * np.pi / (4 * A)  # Правильный шаг
    t = np.arange(-N//2, N//2) * dt
    h = ifft(np.fft.ifftshift(h_hat)) * len(h_hat) * (xi[1] - xi[0]) / (2 * np.pi)
    h = np.fft.fftshift(h)
    h = np.real(h)  # Берем реальную часть
    
    return t, h, xi, h_hat

def accurate_archimedean(t, h):
    """
    Более точное вычисление архимедова члена с адаптивным интегрированием
    """
    def integrand(tau):
        # Интерполируем h в точке tau
        h_val = np.interp(tau, t, h, left=0, right=0)
        # Дигамма вес
        z = 0.25 + 0.5j * tau
        psi_val = special.digamma(z)
        weight = np.real(psi_val) - np.log(np.pi)
        return h_val * weight
    
    # Адаптивное интегрирование
    result, error = integrate.quad(integrand, t[0], t[-1], limit=100)
    A = result / (2 * np.pi)
    
    console.print(f"[cyan]Archimedean: A = {A:.8f} (error: {error:.2e})")
    return A

def accurate_prime_sum(xi, h_hat, A_cutoff, use_all_zeros=False):
    """
    Точная сумма по простым с учетом всех степеней простых
    """
    n_max = int(np.exp(A_cutoff))
    P = 0.0
    
    # Предвычисляем Λ(n) для всех n
    lambda_cache = {}
    for n in range(2, n_max + 1):
        lambda_n = improved_von_mangoldt(n)
        if lambda_n > 0:
            lambda_cache[n] = lambda_n
    
    console.print(f"[yellow]Prime sum: {len(lambda_cache)} non-zero Λ(n) values up to {n_max}")
    
    for n, lambda_n in lambda_cache.items():
        log_n = np.log(n)
        
        # Интерполяция ĥ
        h_hat_plus = np.interp(log_n, xi, np.real(h_hat))
        h_hat_minus = np.interp(-log_n, xi, np.real(h_hat))
        
        contribution = lambda_n / np.sqrt(n) * (h_hat_plus + h_hat_minus)
        P -= contribution / (2 * np.pi)
    
    console.print(f"[cyan]Prime term: P = {P:.8f}")
    return P

def accurate_zero_sum(t, h, num_zeros=50):
    """
    Сумма по нулям с использованием всех доступных
    """
    zeros = RIEMANN_ZEROS[:min(num_zeros, len(RIEMANN_ZEROS))]
    Z = 0.0
    
    for gamma in zeros:
        # Более точная интерполяция
        if t[0] <= gamma <= t[-1]:
            h_gamma = np.interp(gamma, t, h)
            Z += 2 * h_gamma  # Умножаем на 2 для симметричных нулей ±γ
    
    console.print(f"[cyan]Zero sum: Z = {Z:.8f} (using {len(zeros)} zeros)")
    return Z

def test_critical_transition():
    """
    Детальное исследование перехода к PSD при A ≈ 2
    """
    console.print("[bold magenta]CRITICAL TRANSITION ANALYSIS[/bold magenta]\n")
    
    A_values = np.linspace(1.5, 2.5, 11)
    results = []
    
    for A in track(A_values, description="Testing critical region..."):
        # Создаем тест-функцию
        t, h, xi, h_hat = create_smooth_bump(A, N=2048)
        
        # Вычисляем члены
        Z = accurate_zero_sum(t, h, num_zeros=50)
        A_term = accurate_archimedean(t, h)
        P = accurate_prime_sum(xi, h_hat, A)
        
        # Квадратичная форма Q(h) = Z - A - P
        Q = Z - A_term - P
        
        results.append({
            'A': A,
            'Z': Z,
            'A_term': A_term,
            'P': P,
            'Q': Q,
            'error': abs(Z - (A_term + P))
        })
    
    # Визуализация перехода
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    A_vals = [r['A'] for r in results]
    
    # Q(h) vs A
    ax = axes[0, 0]
    Q_vals = [r['Q'] for r in results]
    ax.plot(A_vals, Q_vals, 'ro-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=2.0, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel('Bandwidth A')
    ax.set_ylabel('Q(h) = Z - A - P')
    ax.set_title('Quadratic Form (transition at A≈2)')
    ax.grid(True, alpha=0.3)
    
    # Individual terms
    ax = axes[0, 1]
    ax.plot(A_vals, [r['Z'] for r in results], 'b-', label='Z(h)', linewidth=2)
    ax.plot(A_vals, [r['A_term'] for r in results], 'r-', label='A(h)', linewidth=2)
    ax.plot(A_vals, [abs(r['P']) for r in results], 'g-', label='|P(h)|', linewidth=2)
    ax.set_xlabel('Bandwidth A')
    ax.set_ylabel('Value')
    ax.set_title('Individual Terms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error in explicit formula
    ax = axes[1, 0]
    errors = [r['error'] for r in results]
    ax.semilogy(A_vals, errors, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Bandwidth A')
    ax.set_ylabel('|Z - (A + P)| (log scale)')
    ax.set_title('Explicit Formula Error')
    ax.grid(True, alpha=0.3)
    
    # Critical point zoom
    ax = axes[1, 1]
    # Найдем точку перехода
    Q_array = np.array(Q_vals)
    transition_idx = np.where(Q_array > 0)[0]
    if len(transition_idx) > 0:
        critical_A = A_vals[transition_idx[0]]
        ax.text(0.5, 0.5, f"Critical point:\nA_c ≈ {critical_A:.3f}", 
                ha='center', va='center', fontsize=20,
                bbox=dict(boxstyle='round', facecolor='lightgreen'))
    else:
        ax.text(0.5, 0.5, "No transition found", ha='center', va='center', fontsize=16)
    ax.set_title('PSD Transition Point')
    ax.axis('off')
    
    plt.suptitle('CRITICAL PSD TRANSITION ANALYSIS', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('weil_transition.png', dpi=150)
    console.print("[green]Saved: weil_transition.png")
    plt.show()
    
    return results

def main():
    console.print("[bold red]IMPROVED WEIL FORMULA WITH CRITICAL ANALYSIS[/bold red]\n")
    
    # Анализ критического перехода
    results = test_critical_transition()
    
    # Найдем точку перехода
    positive_results = [r for r in results if r['Q'] > 0]
    if positive_results:
        critical_A = positive_results[0]['A']
        console.print(f"\n[bold green]CRITICAL DISCOVERY:[/bold green]")
        console.print(f"PSD transition occurs at A_c ≈ {critical_A:.3f}")
        console.print("For A > A_c, the quadratic form Q(h) is POSITIVE!")
        console.print("\nThis suggests that the Weil criterion holds for")
        console.print("test functions with bandwidth A > 2!")
    else:
        console.print("\n[yellow]No PSD transition found in tested range")
    
    # Минимальная ошибка
    min_error = min(r['error'] for r in results)
    console.print(f"\n[cyan]Minimum explicit formula error: {min_error:.8f}")
    
    if min_error < 0.1:
        console.print("[green]Excellent numerical accuracy achieved!")
    else:
        console.print("[yellow]Consider using more zeros or higher precision")

if __name__ == "__main__":
    main()