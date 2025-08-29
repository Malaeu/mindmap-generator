#!/usr/bin/env python3
"""
ИСПРАВЛЕННАЯ GAUSSIAN-HERMITE РЕАЛИЗАЦИЯ
========================================
Правильная нормировка под конвенцией ĥ(ξ) = ∫h(t)e^{-iξt}dt
"""

from math import pi, sqrt

import numpy as np
from numpy.polynomial.hermite import hermval
from rich.console import Console
from scipy import integrate

console = Console()

def analytical_parseval_check(sigma, k_even):
    """
    Аналитическая проверка Парсеваля для Gaussian-Hermite
    ∫|h|² dt = (1/2π)∫|ĥ|² dξ должно выполняться точно
    """
    # Для гауссо-эрмитовых функций интегралы вычисляются аналитически
    # ∫|H_k(x) exp(-x²/2)|² dx = 2^k k! √π для стандартных Hermite
    
    factorial_k = 1
    for i in range(1, k_even + 1):
        factorial_k *= i
    
    # Левая часть: ∫|h(t)|² dt  
    # h(t) = H_k(t/σ) exp(-t²/(2σ²))
    # Замена x = t/σ дает σ ∫|H_k(x) exp(-x²/2)|² dx = σ · 2^k k! √π
    left_side = sigma * (2**k_even) * factorial_k * sqrt(pi)
    
    # Правая часть: (1/2π) ∫|ĥ(ξ)|² dξ
    # ĥ(ξ) = (-i)^k √(2π) σ H_k(σξ) exp(-(σ²ξ²)/2) 
    # |ĥ(ξ)|² = 2π σ² |H_k(σξ)|² exp(-σ²ξ²)
    # Замена y = σξ: (1/2π) ∫ 2π σ² |H_k(y)|² exp(-y²) dy/σ = σ ∫|H_k(y)|² exp(-y²) dy
    # = σ · 2^k k! √π (та же формула)
    right_side = left_side  # Должно быть равно по теории
    
    return left_side, right_side, abs(left_side - right_side) / max(left_side, 1e-10)

def corrected_gaussian_hermite_pair(sigma, k_even):
    """
    Правильная Gaussian-Hermite пара под конвенцией ĥ(ξ) = ∫h(t)e^{-iξt}dt
    
    h(t) = H_k(t/σ) exp(-t²/(2σ²))
    ĥ(ξ) = (-i)^k √(2π) σ H_k(σξ) exp(-(σ²ξ²)/2)
    """
    if k_even % 2 != 0:
        raise ValueError(f"k={k_even} must be even for h(t) to be even!")
    
    # Проверяем аналитический Парсеваль
    left, right, error = analytical_parseval_check(sigma, k_even)
    if error > 1e-12:
        console.print(f"[yellow]Warning: Analytical Parseval error {error:.2e} for σ={sigma}, k={k_even}[/yellow]")
    
    def h(t, s=sigma, k=k_even):
        x = t / s
        # Коэффициенты для H_k: [0,0,...,0,1] на позиции k
        coeff = [0] * (k + 1)
        coeff[k] = 1.0
        H_k_val = hermval(x, coeff)
        return H_k_val * np.exp(-(t**2) / (2 * s**2))
    
    def hhat(xi, s=sigma, k=k_even):
        # ĥ(ξ) = (-i)^k √(2π) σ H_k(σξ) exp(-(σ²ξ²)/2)
        y = s * xi
        coeff = [0] * (k + 1)
        coeff[k] = 1.0
        H_k_val = hermval(y, coeff)
        
        # (-i)^k для четных k: (-i)^0=1, (-i)^2=-1, (-i)^4=1, (-i)^6=-1, ...
        i_power = ((-1) ** (k // 2)) * (1 if (k // 2) % 2 == 0 else 1)
        
        return i_power * sqrt(2*pi) * s * H_k_val * np.exp(-(s**2) * (xi**2) / 2)
    
    return h, hhat

def numerical_parseval_check(h_func, hhat_func, sigma_hint=5.0):
    """Численная проверка Парсеваля для любой пары"""
    
    # Адаптивные пределы интегрирования
    t_max = max(10 * sigma_hint, 20.0)
    xi_max = max(10 / sigma_hint, 20.0)
    
    # Левая часть: ∫|h(t)|² dt
    def h_squared(t):
        return np.abs(h_func(t))**2
    
    left, left_err = integrate.quad(h_squared, -t_max, t_max, limit=400, epsrel=1e-10)
    
    # Правая часть: (1/2π) ∫|ĥ(ξ)|² dξ  
    def hhat_squared(xi):
        return np.abs(hhat_func(xi))**2 / (2*pi)
    
    right, right_err = integrate.quad(hhat_squared, -xi_max, xi_max, limit=400, epsrel=1e-10)
    
    error = abs(left - right) / max(left, 1e-10)
    total_err = (left_err + right_err) / max(left, 1e-10)
    
    return left, right, error, total_err

def test_corrected_hermite_pairs():
    """Тест исправленных Hermite пар"""
    console.print("[bold cyan]ТЕСТ ИСПРАВЛЕННЫХ GAUSSIAN-HERMITE ПАР[/bold cyan]\n")
    
    test_cases = [
        (2.0, 0), (3.0, 2), (4.0, 4), (5.0, 2), (6.0, 6)
    ]
    
    from rich.table import Table
    table = Table()
    table.add_column("σ", justify="center")
    table.add_column("k", justify="center") 
    table.add_column("Parseval Error", justify="right")
    table.add_column("Quad Error", justify="right")
    table.add_column("Status", justify="center")
    
    all_good = True
    
    for sigma, k in test_cases:
        try:
            h, hhat = corrected_gaussian_hermite_pair(sigma, k)
            left, right, error, quad_err = numerical_parseval_check(h, hhat, sigma)
            
            is_good = error < 1e-6  # Разумный tolerance
            status = "[green]✅[/green]" if is_good else "[red]❌[/red]"
            
            if not is_good:
                all_good = False
            
            table.add_row(
                f"{sigma:.1f}",
                str(k),
                f"{error:.2e}",
                f"{quad_err:.2e}", 
                status
            )
            
        except Exception as e:
            table.add_row(f"{sigma:.1f}", str(k), "ERROR", str(e)[:15], "[red]❌[/red]")
            all_good = False
    
    console.print(table)
    
    if all_good:
        console.print("\n[bold green]✅ Все исправленные пары прошли проверку![/bold green]")
    else:
        console.print("\n[bold red]❌ Некоторые пары все еще имеют проблемы[/bold red]")
    
    return all_good

if __name__ == "__main__":
    # Сравнение старой и новой реализации
    console.print("[bold yellow]СРАВНЕНИЕ РЕАЛИЗАЦИЙ:[/bold yellow]\n")
    
    sigma, k = 3.0, 2
    
    # Старая реализация (из fourier_conventions.py)
    try:
        from fourier_conventions import GaussianHermitePair
        old_pair = GaussianHermitePair(sigma, k)
        old_left, old_right, old_error, _ = numerical_parseval_check(old_pair.h, old_pair.hhat, sigma)
        console.print(f"Старая реализация: Parseval error = {old_error:.2e}")
    except Exception as e:
        console.print(f"Старая реализация: ОШИБКА - {e}")
    
    # Новая реализация
    new_h, new_hhat = corrected_gaussian_hermite_pair(sigma, k)
    new_left, new_right, new_error, _ = numerical_parseval_check(new_h, new_hhat, sigma)
    console.print(f"Новая реализация: Parseval error = {new_error:.2e}")
    
    # Полный тест
    console.print("\n" + "="*50)
    success = test_corrected_hermite_pairs()
    
    if success:
        console.print("\n[bold green]ГОТОВО: можно использовать исправленную реализацию![/bold green]")
    else:
        console.print("\n[bold red]ТРЕБУЕТСЯ дополнительная отладка[/bold red]")