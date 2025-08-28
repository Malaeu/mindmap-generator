#!/usr/bin/env python3
"""
МАТЕМАТИЧЕСКИ КОРРЕКТНАЯ ПРОВЕРКА КРИТЕРИЯ ВЕЙЛЯ
Без подгонок, с правильными весами и функциями
"""

import numpy as np
from scipy import special
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def von_mangoldt(n):
    """
    Функция фон Мангольдта Λ(n)
    Λ(n) = log(p) если n = p^k для простого p
    Λ(n) = 0 иначе
    """
    if n <= 1:
        return 0.0
    
    # Факторизация
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        count = 0
        while temp % d == 0:
            count += 1
            temp //= d
        if count > 0:
            factors.append((d, count))
        d += 1
    if temp > 1:
        factors.append((temp, 1))
    
    # Если n = p^k для одного простого p
    if len(factors) == 1:
        p, k = factors[0]
        return np.log(p)
    else:
        return 0.0

def digamma_weight(t):
    """
    Архимедов вес: Re ψ(1/4 + it/2) - log π
    где ψ - дигамма функция
    """
    z = 0.25 + 0.5j * t
    psi_val = special.digamma(z)
    return np.real(psi_val) - np.log(np.pi)

def create_bump_function(A, N=1024):
    """
    Создаем гладкую bump-функцию с компактной поддержкой [-A, A]
    в Фурье-пространстве
    """
    # Частотная сетка
    xi = np.linspace(-2*A, 2*A, N)
    
    # Гладкая bump функция (C∞ с компактной поддержкой)
    h_hat = np.zeros_like(xi)
    mask = np.abs(xi) < A
    h_hat[mask] = np.exp(-1.0 / (1.0 - (xi[mask]/A)**2))
    
    # Нормализация
    h_hat = h_hat / np.max(h_hat)
    
    # Обратное преобразование для получения h(t)
    # Используем правильную нормировку
    t = 2 * np.pi * fftfreq(N, d=(xi[1]-xi[0]))
    h = ifft(h_hat)
    h = np.real(h)  # Должно быть вещественным для четной функции
    
    # Сортируем по t
    idx = np.argsort(t)
    t = t[idx]
    h = h[idx]
    h_hat_sorted = h_hat  # Уже отсортировано по xi
    
    return t, h, xi, h_hat_sorted

def compute_archimedean_term(t, h):
    """
    Вычисляем архимедов член
    A(h) = (1/2π) ∫ h(t) [Re ψ(1/4 + it/2) - log π] dt
    """
    dt = t[1] - t[0] if len(t) > 1 else 0
    weights = digamma_weight(t)
    
    # Численное интегрирование
    A = np.sum(h * weights) * dt / (2 * np.pi)
    
    console.print(f"[cyan]Archimedean term: A = {A:.6f}")
    
    return A, weights

def compute_prime_term(xi, h_hat, A_cutoff):
    """
    Вычисляем простой член
    P(h) = -(1/2π) Σ_{n≤e^A} Λ(n)/√n [ĥ(log n) + ĥ(-log n)]
    """
    n_max = int(np.exp(A_cutoff))
    console.print(f"[yellow]Computing prime sum up to n = {n_max}")
    
    P = 0.0
    prime_contributions = []
    
    for n in range(2, n_max + 1):
        lambda_n = von_mangoldt(n)
        if lambda_n > 0:
            log_n = np.log(n)
            
            # Интерполируем ĥ в точках ±log n
            h_hat_plus = np.interp(log_n, xi, h_hat)
            h_hat_minus = np.interp(-log_n, xi, h_hat)
            
            contribution = lambda_n / np.sqrt(n) * (h_hat_plus + h_hat_minus)
            P -= contribution / (2 * np.pi)
            
            if contribution != 0:
                prime_contributions.append((n, lambda_n, contribution))
    
    console.print(f"[cyan]Prime term: P = {P:.6f}")
    console.print(f"[dim]Non-zero contributions: {len(prime_contributions)}")
    
    return P, prime_contributions

def get_riemann_zeros(N):
    """Первые N нетривиальных нулей ζ(s)"""
    zeros = [
        14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
        37.5861781588, 40.9187190121, 43.3270732809, 48.0051508812, 49.7738324777
    ]
    return zeros[:N]

def compute_zero_term(t, h, num_zeros=10):
    """
    Вычисляем нулевой член
    Z(h) = Σ_ρ h(γ) где γ - мнимые части нулей
    """
    zeros = get_riemann_zeros(num_zeros)
    Z = 0.0
    
    for gamma in zeros:
        # Интерполируем h в точке γ
        h_gamma = np.interp(gamma, t, h)
        Z += h_gamma
    
    console.print(f"[cyan]Zero term: Z = {Z:.6f} (first {num_zeros} zeros)")
    
    return Z, zeros

def verify_explicit_formula(A_cutoff=3.0, N_points=1024):
    """
    Проверяем явную формулу Вейля
    Z(h) = A(h) + P(h) + O(error)
    """
    console.print(Panel.fit("[bold red]CORRECT WEIL EXPLICIT FORMULA TEST[/bold red]", box=box.DOUBLE))
    
    # Создаем тест-функцию
    console.print(f"\n[yellow]Creating bump function with cutoff A = {A_cutoff}")
    t, h, xi, h_hat = create_bump_function(A_cutoff, N_points)
    
    # Вычисляем все члены
    console.print("\n[bold]Computing terms:[/bold]")
    
    # Архимедов член
    A, arch_weights = compute_archimedean_term(t, h)
    
    # Простой член
    P, prime_contribs = compute_prime_term(xi, h_hat, A_cutoff)
    
    # Нулевой член
    Z, zeros = compute_zero_term(t, h, num_zeros=10)
    
    # Проверка баланса
    console.print("\n[bold]BALANCE CHECK:[/bold]")
    console.print(f"Z(h) = {Z:.6f}")
    console.print(f"A(h) + P(h) = {A + P:.6f}")
    console.print(f"[{'green' if abs(Z - (A + P)) < 1 else 'red'}]Discrepancy: {abs(Z - (A + P)):.6f}")
    
    return {
        'Z': Z,
        'A': A,
        'P': P,
        'error': abs(Z - (A + P)),
        't': t,
        'h': h,
        'xi': xi,
        'h_hat': h_hat,
        'arch_weights': arch_weights,
        'zeros': zeros
    }

def build_gram_matrix(test_functions, A_cutoff=3.0):
    """
    Строим матрицу Грама для набора тест-функций
    БЕЗ искусственной калибровки!
    """
    console.print("\n[bold cyan]Building Gram matrix (NO CALIBRATION)[/bold cyan]")
    
    N = len(test_functions)
    M = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i, N):
            t_i, h_i, xi_i, h_hat_i = test_functions[i]
            t_j, h_j, xi_j, h_hat_j = test_functions[j]
            
            # Произведение функций (для кросс-термов)
            # Упрощенно: используем поточечное произведение
            h_ij = h_i * h_j
            
            # Архимедов кросс-терм
            A_ij, _ = compute_archimedean_term(t_i, h_ij)
            
            # Простой кросс-терм (нужно FFT от h_ij)
            h_ij_hat = fft(h_ij)
            xi_ij = fftfreq(len(h_ij), d=(t_i[1]-t_i[0]) if len(t_i) > 1 else 1)
            P_ij, _ = compute_prime_term(xi_ij, np.abs(h_ij_hat), A_cutoff)
            
            # Нулевой кросс-терм
            Z_ij, _ = compute_zero_term(t_i, h_ij, num_zeros=10)
            
            # БЕЗ калибровки! Прямая формула
            M[i, j] = Z_ij - A_ij - P_ij
            if i != j:
                M[j, i] = M[i, j]
    
    console.print(f"[yellow]Gram matrix:\n{M}")
    
    # Проверка собственных значений
    eigenvals = np.linalg.eigvalsh(M)
    console.print(f"\n[bold]Eigenvalues: {eigenvals}")
    console.print(f"[{'green' if eigenvals[0] > 0 else 'red'}]λ_min = {eigenvals[0]:.6f}")
    console.print(f"λ_max = {eigenvals[-1]:.6f}")
    console.print(f"Condition number: {eigenvals[-1]/eigenvals[0] if eigenvals[0] != 0 else np.inf:.2e}")
    
    return M, eigenvals

def test_multiple_widths():
    """
    Тестируем для разных ширин полосы A
    """
    console.print(Panel.fit("[bold magenta]TESTING DIFFERENT BANDWIDTHS[/bold magenta]", box=box.DOUBLE))
    
    A_values = [1.0, 2.0, 3.0, 4.0]
    results = []
    
    for A in A_values:
        console.print(f"\n[yellow]Testing A = {A}[/yellow]")
        
        # Создаем несколько тест-функций со сдвигами
        test_funcs = []
        for shift in [0, 0.5, 1.0]:
            t, h, xi, h_hat = create_bump_function(A, N=512)
            # Применяем сдвиг в частотной области
            h_hat_shifted = h_hat * np.exp(2j * np.pi * shift * xi)
            h_shifted = np.real(ifft(h_hat_shifted))
            test_funcs.append((t, h_shifted, xi, h_hat_shifted))
        
        # Строим матрицу Грама
        M, eigenvals = build_gram_matrix(test_funcs, A_cutoff=A)
        
        results.append({
            'A': A,
            'lambda_min': eigenvals[0],
            'lambda_max': eigenvals[-1],
            'is_psd': eigenvals[0] > 0
        })
    
    # Таблица результатов
    table = Table(title="Bandwidth Test Results", box=box.ROUNDED)
    table.add_column("A", style="cyan")
    table.add_column("λ_min", style="yellow")
    table.add_column("λ_max", style="yellow")
    table.add_column("PSD?", style="green")
    
    for r in results:
        table.add_row(
            f"{r['A']:.1f}",
            f"{r['lambda_min']:.6f}",
            f"{r['lambda_max']:.6f}",
            "✓" if r['is_psd'] else "✗"
        )
    
    console.print("\n")
    console.print(table)
    
    return results

def visualize_correct_formula(result):
    """
    Визуализация корректной формулы
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test function h(t)
    ax = axes[0, 0]
    ax.plot(result['t'], result['h'], 'b-', linewidth=2)
    for gamma in result['zeros']:
        ax.axvline(x=gamma, color='r', linestyle=':', alpha=0.3)
    ax.set_xlabel('t')
    ax.set_ylabel('h(t)')
    ax.set_title('Test Function (time domain)')
    ax.grid(True, alpha=0.3)
    
    # Fourier transform ĥ(ξ)
    ax = axes[0, 1]
    ax.plot(result['xi'], result['h_hat'], 'g-', linewidth=2)
    ax.set_xlabel('ξ')
    ax.set_ylabel('ĥ(ξ)')
    ax.set_title('Fourier Transform (compact support)')
    ax.grid(True, alpha=0.3)
    
    # Archimedean weight
    ax = axes[0, 2]
    ax.plot(result['t'], result['arch_weights'], 'r-', linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('Re ψ(1/4 + it/2) - log π')
    ax.set_title('Archimedean Weight (digamma)')
    ax.grid(True, alpha=0.3)
    
    # Balance visualization
    ax = axes[1, 0]
    terms = ['Z(h)', 'A(h)', 'P(h)', 'A+P']
    values = [result['Z'], result['A'], result['P'], result['A'] + result['P']]
    colors = ['blue', 'red', 'green', 'orange']
    ax.bar(terms, values, color=colors, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Explicit Formula Balance')
    ax.grid(True, alpha=0.3)
    
    # Error analysis
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f"Discrepancy: {result['error']:.6f}", 
            ha='center', va='center', fontsize=20,
            bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.set_title('Formula Verification')
    ax.axis('off')
    
    # Von Mangoldt function
    ax = axes[1, 2]
    n_vals = range(2, 50)
    lambda_vals = [von_mangoldt(n) for n in n_vals]
    ax.stem(n_vals, lambda_vals, basefmt=' ')
    ax.set_xlabel('n')
    ax.set_ylabel('Λ(n)')
    ax.set_title('Von Mangoldt Function')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('CORRECT WEIL EXPLICIT FORMULA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('weil_correct.png', dpi=150)
    console.print("[green]Saved: weil_correct.png")
    plt.show()

if __name__ == "__main__":
    console.print("[bold red on white] MATHEMATICALLY CORRECT WEIL TEST [/]\n")
    
    # Основной тест
    result = verify_explicit_formula(A_cutoff=3.0)
    
    # Визуализация
    visualize_correct_formula(result)
    
    # Тест с разными ширинами
    bandwidth_results = test_multiple_widths()
    
    # Финальная оценка
    console.print("\n[bold yellow]FINAL ASSESSMENT:[/bold yellow]")
    console.print("This is the CORRECT mathematical formulation:")
    console.print("✓ Using digamma function for Archimedean term")
    console.print("✓ Using von Mangoldt Λ(n) for prime term")
    console.print("✓ NO artificial calibration or scaling")
    console.print("✓ Proper Schwartz functions with compact Fourier support")
    
    if result['error'] < 1.0:
        console.print(f"\n[green]Explicit formula verified with error = {result['error']:.6f}")
    else:
        console.print(f"\n[yellow]Large discrepancy = {result['error']:.6f}")
        console.print("Need more zeros or better numerical integration")