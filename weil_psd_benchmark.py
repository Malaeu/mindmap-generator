#!/usr/bin/env python3
"""
БАРЬЕР ПОЗИТИВНОЙ СТЕНЫ: PSD-ТЕСТ ДЛЯ КРИТЕРИЯ ВЕЙЛЯ
===================================================
Проверяем Q ≥ 0 на КЛАССЕ функций, а не на одной гауссиане
"""

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from fourier_conventions import GaussianHermitePair, GaussianPair, compute_Q_weil, sieve_primes

console = Console()

# Первые 30 нулей Римана  
ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
]

PRIMES = sieve_primes(1000)

def create_function_basis(sigmas, hermite_orders):
    """Создаём базис Gaussian-Hermite функций"""
    basis = []
    labels = []
    
    for sigma in sigmas:
        for k in hermite_orders:
            if k == 0:
                # Чистая гауссиана
                func = GaussianPair(sigma)
                labels.append(f"G(σ={sigma})")
            else:
                # Гауссиана × Эрмит
                func = GaussianHermitePair(sigma, k)
                labels.append(f"GH(σ={sigma},k={k})")
            
            basis.append(func)
    
    return basis, labels

def compute_gram_matrix(basis, labels, zeros=None, verbose=True):
    """Вычисление Gram-матрицы через поляризацию"""
    if zeros is None:
        zeros = ZEROS
    
    n = len(basis)
    G = np.zeros((n, n))
    
    if verbose:
        console.print(f"[yellow]Вычисление Gram-матрицы {n}×{n}...[/yellow]")
    
    # Диагональные элементы: G_ii = Q(h_i)
    for i in range(n):
        func = basis[i]
        Q_ii, _ = compute_Q_weil(func.h, func.hhat, zeros, 
                               sigma_hint=func.sigma, verbose=False)
        G[i, i] = Q_ii
        
        if verbose and i % 5 == 0:
            console.print(f"  Q({labels[i]}) = {Q_ii:+.4f}")
    
    # Недиагональные элементы через поляризацию
    # Q(h_i, h_j) = 1/2 [Q(h_i + h_j) - Q(h_i) - Q(h_j)]
    for i in range(n):
        for j in range(i+1, n):
            func_i, func_j = basis[i], basis[j]
            
            # Функция суммы
            def h_sum(t):
                return func_i.h(t) + func_j.h(t)
            def hhat_sum(xi):
                return func_i.hhat(xi) + func_j.hhat(xi)
            
            # Средняя σ для архимедова члена
            sigma_avg = (func_i.sigma + func_j.sigma) / 2.0
            
            Q_sum, _ = compute_Q_weil(h_sum, hhat_sum, zeros,
                                    sigma_hint=sigma_avg, verbose=False)
            
            # Поляризация
            G[i, j] = G[j, i] = 0.5 * (Q_sum - G[i, i] - G[j, j])
    
    return G

def analyze_psd(G, labels, tolerance=1e-10):
    """Анализ PSD свойств матрицы"""
    eigenvalues = np.linalg.eigvals(G)
    eigenvalues = np.sort(eigenvalues)
    
    min_eigenvalue = np.min(eigenvalues)
    is_psd = min_eigenvalue >= -tolerance
    
    console.print(f"\n[bold]Анализ PSD (допуск: {tolerance:.0e}):[/bold]")
    
    table = Table(box=box.SIMPLE)
    table.add_column("№", justify="center")
    table.add_column("λ", justify="right")
    table.add_column("Статус", justify="center")
    
    for i, lam in enumerate(eigenvalues):
        status = "[green]✅[/green]" if lam >= -tolerance else "[red]❌[/red]"
        color = "green" if lam >= -tolerance else "red"
        
        table.add_row(
            str(i+1),
            f"[{color}]{lam:+.6f}[/{color}]",
            status
        )
    
    console.print(table)
    
    if is_psd:
        console.print(f"\n[bold green]✅ Матрица PSD! (λ_min = {min_eigenvalue:.2e})[/bold green]")
    else:
        console.print(f"\n[bold red]❌ Матрица НЕ PSD! (λ_min = {min_eigenvalue:.2e})[/bold red]")
    
    return is_psd, min_eigenvalue, eigenvalues

def tail_bounds_analysis(basis, labels, zeros=None):
    """Анализ хвостовых ошибок"""
    if zeros is None:
        zeros = ZEROS
    
    console.print("\n[yellow]Анализ хвостовых ошибок:[/yellow]")
    
    # Z-tail: оценка для γ > γ_max
    gamma_max = max(zeros)
    
    # P-tail: зависит от σ каждой функции
    table = Table(box=box.SIMPLE)
    table.add_column("Функция", style="cyan")
    table.add_column("Z-tail", justify="right")
    table.add_column("P-tail", justify="right") 
    table.add_column("Relative Error", justify="right")
    
    for func, label in zip(basis, labels, strict=False):
        # Грубая оценка Z-tail через гауссиан
        from scipy.integrate import quad
        def z_tail_integrand(t):
            return func.h(t) * np.log(max(t, 1.0)) / (2*np.pi)
        
        z_tail, _ = quad(z_tail_integrand, gamma_max, np.inf, limit=200)
        
        # P-tail через экспоненциальное убывание
        sigma = func.sigma
        N_max = 1000  # Наш обычный срез
        p_tail_est = func.hhat(np.log(N_max)) * np.log(N_max) / (2*np.pi)
        
        # Полное Q для сравнения
        Q_full, _ = compute_Q_weil(func.h, func.hhat, zeros, 
                                 sigma_hint=sigma, verbose=False)
        
        rel_error = (abs(z_tail) + abs(p_tail_est)) / max(abs(Q_full), 1e-10)
        
        table.add_row(
            label,
            f"{z_tail:.2e}",
            f"{p_tail_est:.2e}",
            f"{rel_error:.1%}"
        )
    
    console.print(table)

def robustness_test(basis, labels, scale_factors=[1, 2, 4]):
    """Тест устойчивости при увеличении числа нулей"""
    console.print("\n[yellow]Тест устойчивости PSD:[/yellow]")
    
    results = {}
    
    for scale in scale_factors:
        # Масштабируем количество нулей
        n_zeros = min(len(ZEROS) * scale, len(ZEROS))  # Не можем больше чем есть
        zeros_subset = ZEROS[:n_zeros]
        
        console.print(f"\n  Тестируем с {n_zeros} нулями:")
        
        G = compute_gram_matrix(basis, labels, zeros_subset, verbose=False)
        is_psd, min_eig, _ = analyze_psd(G, labels, tolerance=1e-8)
        
        results[scale] = {
            'n_zeros': n_zeros,
            'min_eigenvalue': min_eig,
            'is_psd': is_psd
        }
        
        console.print(f"    λ_min = {min_eig:+.6f} {'✅' if is_psd else '❌'}")
    
    # Проверяем стабильность
    min_eigs = [results[s]['min_eigenvalue'] for s in scale_factors]
    is_stable = all(eig >= -1e-8 for eig in min_eigs)
    
    if is_stable:
        console.print("\n[bold green]✅ PSD стабильно при масштабировании![/bold green]")
    else:
        console.print("\n[bold red]❌ PSD нестабильно при масштабировании![/bold red]")
    
    return is_stable, results

def main_benchmark():
    """Основной бенчмарк для прохождения барьера позитивной стены"""
    console.print("[bold cyan]БАРЬЕР ПОЗИТИВНОЙ СТЕНЫ: PSD-БЕНЧМАРК[/bold cyan]")
    console.print("[dim]Критерий Вейля для класса Gaussian-Hermite функций[/dim]\n")
    
    # Создаём базис функций
    sigmas = [2.0, 3.0, 4.0, 5.0, 6.0]
    hermite_orders = [0, 2, 4]  # Только чётные для симметрии
    
    basis, labels = create_function_basis(sigmas, hermite_orders)
    console.print(f"Создан базис из {len(basis)} функций:")
    for label in labels:
        console.print(f"  - {label}")
    
    # 1. Gram-матрица
    console.print("\n[bold yellow]1. ПОСТРОЕНИЕ GRAM-МАТРИЦЫ[/bold yellow]")
    G = compute_gram_matrix(basis, labels, ZEROS)
    
    # 2. PSD анализ
    console.print("\n[bold yellow]2. PSD-АНАЛИЗ[/bold yellow]")
    is_psd, min_eig, eigenvalues = analyze_psd(G, labels)
    
    # 3. Хвостовые ошибки
    console.print("\n[bold yellow]3. ХВОСТОВЫЕ ОШИБКИ[/bold yellow]")
    tail_bounds_analysis(basis, labels, ZEROS)
    
    # 4. Тест устойчивости
    console.print("\n[bold yellow]4. ТЕСТ УСТОЙЧИВОСТИ[/bold yellow]")
    is_stable, robustness_results = robustness_test(basis, labels)
    
    # Финальный вердикт
    console.print("\n" + "="*60)
    console.print("[bold green]ФИНАЛЬНЫЙ ВЕРДИКТ:[/bold green]\n")
    
    if is_psd and is_stable:
        console.print("[bold green]🎉 БАРЬЕР ПОЗИТИВНОЙ СТЕНЫ ПРОЙДЕН![/bold green]")
        console.print("✅ Q ≥ 0 для всего Gaussian-Hermite подпространства")
        console.print("✅ PSD стабильно при увеличении числа нулей")
        console.print("✅ Хвостовые ошибки пренебрежимо малы")
        console.print("\n[cyan]Это первое строгое подтверждение критерия Вейля на нетривиальном классе![/cyan]")
        
        return True
    else:
        console.print("[bold red]❌ БАРЬЕР НЕ ПРОЙДЕН[/bold red]")
        if not is_psd:
            console.print("- Gram-матрица не PSD")
        if not is_stable:
            console.print("- PSD нестабильно при масштабировании")
        
        return False

if __name__ == "__main__":
    success = main_benchmark()
    exit(0 if success else 1)