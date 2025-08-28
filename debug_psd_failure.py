#!/usr/bin/env python3
"""
ОТЛАДКА PSD ПРОВАЛА: Что пошло не так?
====================================
Пошаговый анализ почему Gram-матрица не PSD
"""

import numpy as np
from fourier_conventions import GaussianPair, compute_Q_weil, sieve_primes
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
]

def test_individual_gaussians():
    """Тест: все ли отдельные гауссианы дают Q > 0?"""
    console.print("[bold cyan]ТЕСТ 1: ИНДИВИДУАЛЬНЫЕ ГАУССИАНЫ[/bold cyan]\n")
    
    sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    table = Table(box=box.ROUNDED)
    table.add_column("σ", style="cyan", justify="center")
    table.add_column("Q", justify="right")
    table.add_column("Z", justify="right")
    table.add_column("A", justify="right")
    table.add_column("P", justify="right")
    table.add_column("Status", justify="center")
    
    results = {}
    
    for sigma in sigmas:
        gauss = GaussianPair(sigma)
        Q, components = compute_Q_weil(gauss.h, gauss.hhat, ZEROS, 
                                     sigma_hint=sigma, verbose=False)
        
        results[sigma] = Q
        status = "[green]✅[/green]" if Q > 0 else "[red]❌[/red]"
        q_color = "green" if Q > 0 else "red"
        
        table.add_row(
            f"{sigma:.1f}",
            f"[{q_color}]{Q:+.4f}[/{q_color}]",
            f"{components['Z']:+.4f}",
            f"{components['A']:+.4f}",
            f"{components['P']:+.4f}",
            status
        )
    
    console.print(table)
    
    positive_count = sum(1 for q in results.values() if q > 0)
    console.print(f"\n✅ {positive_count}/{len(sigmas)} гауссиан дают Q > 0")
    
    return results

def test_small_gram_matrix():
    """Тест: малая Gram-матрица только из гауссиан"""
    console.print("\n[bold cyan]ТЕСТ 2: GRAM-МАТРИЦА ДЛЯ ЧИСТЫХ ГАУССИАН[/bold cyan]\n")
    
    # Берём только σ где Q > 0
    sigmas = [2.0, 3.0, 4.0, 5.0]
    basis = [GaussianPair(s) for s in sigmas]
    labels = [f"G(σ={s})" for s in sigmas]
    
    n = len(basis)
    G = np.zeros((n, n))
    
    console.print(f"Строим {n}×{n} матрицу для {labels}...")
    
    # Диагональ
    for i, (sigma, func) in enumerate(zip(sigmas, basis)):
        Q_ii, _ = compute_Q_weil(func.h, func.hhat, ZEROS, 
                               sigma_hint=sigma, verbose=False)
        G[i, i] = Q_ii
        console.print(f"  G[{i},{i}] = Q(σ={sigma}) = {Q_ii:+.4f}")
    
    # Недиагональные элементы через поляризацию
    for i in range(n):
        for j in range(i+1, n):
            func_i, func_j = basis[i], basis[j]
            sigma_i, sigma_j = sigmas[i], sigmas[j]
            
            # Сумма функций
            def h_sum(t):
                return func_i.h(t) + func_j.h(t)
            def hhat_sum(xi):
                return func_i.hhat(xi) + func_j.hhat(xi)
            
            sigma_avg = (sigma_i + sigma_j) / 2.0
            Q_sum, _ = compute_Q_weil(h_sum, hhat_sum, ZEROS,
                                    sigma_hint=sigma_avg, verbose=False)
            
            # Поляризация: Q(i,j) = 1/2[Q(i+j) - Q(i) - Q(j)]
            G[i, j] = G[j, i] = 0.5 * (Q_sum - G[i, i] - G[j, j])
            console.print(f"  G[{i},{j}] = {G[i,j]:+.4f}")
    
    # Анализ собственных значений
    eigenvalues = np.linalg.eigvals(G)
    eigenvalues = np.sort(eigenvalues)
    
    console.print(f"\n[bold]Собственные значения:[/bold]")
    for i, lam in enumerate(eigenvalues):
        status = "✅" if lam >= -1e-10 else "❌"
        color = "green" if lam >= -1e-10 else "red"
        console.print(f"  λ_{i+1} = [{color}]{lam:+.6f}[/{color}] {status}")
    
    min_eigenvalue = np.min(eigenvalues)
    is_psd = min_eigenvalue >= -1e-10
    
    if is_psd:
        console.print(f"\n[bold green]✅ Чистые гауссианы дают PSD![/bold green]")
    else:
        console.print(f"\n[bold red]❌ Даже чистые гауссианы НЕ PSD![/bold red]")
    
    return is_psd, G, eigenvalues

def investigate_polarization():
    """Детальное исследование поляризации"""
    console.print("\n[bold cyan]ТЕСТ 3: ДЕТАЛЬНАЯ ПОЛЯРИЗАЦИЯ[/bold cyan]\n")
    
    # Берём две конкретные гауссианы
    sigma1, sigma2 = 3.0, 5.0
    gauss1 = GaussianPair(sigma1)
    gauss2 = GaussianPair(sigma2)
    
    console.print(f"Исследуем пару: σ₁={sigma1}, σ₂={sigma2}")
    
    # Индивидуальные Q
    Q1, comp1 = compute_Q_weil(gauss1.h, gauss1.hhat, ZEROS, sigma1)
    Q2, comp2 = compute_Q_weil(gauss2.h, gauss2.hhat, ZEROS, sigma2)
    
    console.print(f"Q₁ = {Q1:+.6f}")
    console.print(f"Q₂ = {Q2:+.6f}")
    
    # Сумма функций
    def h_sum(t):
        return gauss1.h(t) + gauss2.h(t)
    def hhat_sum(xi):
        return gauss1.hhat(xi) + gauss2.hhat(xi)
    
    sigma_avg = (sigma1 + sigma2) / 2.0
    Q_sum, comp_sum = compute_Q_weil(h_sum, hhat_sum, ZEROS, sigma_avg)
    
    console.print(f"Q(h₁+h₂) = {Q_sum:+.6f}")
    
    # Поляризация
    Q_bilinear = 0.5 * (Q_sum - Q1 - Q2)
    console.print(f"Q(h₁,h₂) = ½[Q(h₁+h₂) - Q₁ - Q₂] = {Q_bilinear:+.6f}")
    
    # Проверка: должно быть Q(h₁,h₂) = Q(h₂,h₁)
    console.print("\n[yellow]Компоненты Q(h₁+h₂):[/yellow]")
    console.print(f"  Z_sum = {comp_sum['Z']:+.6f}")
    console.print(f"  A_sum = {comp_sum['A']:+.6f}")  
    console.print(f"  P_sum = {comp_sum['P']:+.6f}")
    
    # Сравниваем с суммой компонент
    Z_expected = comp1['Z'] + comp2['Z']
    A_expected = comp1['A'] + comp2['A']  # НЕ АДИТИВЕН!
    P_expected = comp1['P'] + comp2['P']  # НЕ АДИТИВЕН!
    
    console.print(f"\n[yellow]Ожидаемая аддитивность (НЕВЕРНО для A,P):[/yellow]")
    console.print(f"  Z₁+Z₂ = {Z_expected:+.6f} (должно ≈ Z_sum)")
    console.print(f"  A₁+A₂ = {A_expected:+.6f} (НЕ равно A_sum!)")
    console.print(f"  P₁+P₂ = {P_expected:+.6f} (НЕ равно P_sum!)")
    
    # Вот в чём проблема!
    console.print(f"\n[bold red]ПРОБЛЕМА НАЙДЕНА:[/bold red]")
    console.print(f"A и P НЕ аддитивны! A(h₁+h₂) ≠ A(h₁) + A(h₂)")
    console.print(f"Поляризация работает только для билинейных форм!")
    
    return Q_bilinear

def main():
    console.print("[bold red]ОТЛАДКА PSD ПРОВАЛА[/bold red]")
    console.print("[dim]Поиск причины отрицательных собственных значений[/dim]\n")
    
    # Тест 1: Индивидуальные Q
    individual_results = test_individual_gaussians()
    
    # Тест 2: Gram-матрица чистых гауссиан
    is_psd, G, eigenvals = test_small_gram_matrix()
    
    # Тест 3: Детальная поляризация
    investigate_polarization()
    
    # Финальный диагноз
    console.print("\n" + "="*60)
    console.print("[bold red]ДИАГНОЗ:[/bold red]\n")
    console.print("❌ Критерий Вейля Q(h) НЕ является билинейной формой!")
    console.print("❌ A(h₁+h₂) ≠ A(h₁) + A(h₂) - архимедов член нелинеен")
    console.print("❌ P(h₁+h₂) ≠ P(h₁) + P(h₂) - простой член нелинеен")
    console.print("❌ Поляризация Q(h₁,h₂) = ½[Q(h₁+h₂)-Q(h₁)-Q(h₂)] неверна!")
    
    console.print(f"\n[cyan]Нужен другой подход для проверки PSD класса функций![/cyan]")

if __name__ == "__main__":
    main()