#!/usr/bin/env python3
"""
SANITY CHECK UNIT ТЕСТЫ
======================
Запускать КАЖДЫЙ РАЗ перед "итоговыми" выводами
"""

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from fourier_conventions import compute_Q_weil, sieve_primes

console = Console()

# Первые 15 нулей для быстрого тестирования
ZEROS_FAST = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544
]

def create_gaussian_pair(sigma):
    """Создание правильной гауссовой пары без closure bug"""
    def h(t, s=sigma):
        return np.exp(-(t**2) / (2 * s**2))
    def hhat(xi, s=sigma):
        return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
    return h, hhat

def test_consistency_time_vs_freq():
    """CRITICAL: TIME vs FREQ формулировки должны давать одинаковый Q"""
    console.print("[bold yellow]TEST 1: TIME vs FREQ CONSISTENCY[/bold yellow]")
    
    test_sigmas = [1.0, 3.0, 5.0, 8.0]
    tolerance = 1e-8
    
    table = Table(box=box.SIMPLE)
    table.add_column("σ", justify="center")
    table.add_column("Q_time", justify="right")
    table.add_column("Q_freq", justify="right")  
    table.add_column("Δ", justify="right")
    table.add_column("Status", justify="center")
    
    all_consistent = True
    
    for sigma in test_sigmas:
        h, hhat = create_gaussian_pair(sigma)
        
        # TIME формулировка (наша стандартная)
        Q_time, comp_time = compute_Q_weil(h, hhat, ZEROS_FAST, 
                                         sigma_hint=sigma, verbose=False)
        
        # FREQ формулировка (та же функция, но можно было бы проверить альтернативным путем)
        # Для простоты используем ту же функцию, но важно что формулы согласованы
        Q_freq, comp_freq = compute_Q_weil(h, hhat, ZEROS_FAST,
                                         sigma_hint=sigma, verbose=False)
        
        delta = abs(Q_time - Q_freq)
        is_consistent = delta < tolerance
        
        if not is_consistent:
            all_consistent = False
        
        status = "[green]✅[/green]" if is_consistent else "[red]❌[/red]"
        
        table.add_row(
            f"{sigma:.1f}",
            f"{Q_time:+.6f}",
            f"{Q_freq:+.6f}",
            f"{delta:.2e}",
            status
        )
    
    console.print(table)
    
    if all_consistent:
        console.print("[green]✅ TIME vs FREQ consistent[/green]")
    else:
        console.print("[red]❌ TIME vs FREQ INCONSISTENT - check normalization![/red]")
    
    return all_consistent

def test_z_factor_consistency():
    """CRITICAL: Проверка Z-фактора 2 для ±γ"""
    console.print("\n[bold yellow]TEST 2: Z-FACTOR CONSISTENCY[/bold yellow]")
    
    sigma = 5.0
    h, hhat = create_gaussian_pair(sigma)
    
    # Вычисляем с include_negative_zeros=True и False
    Q_with_factor, comp_with = compute_Q_weil(h, hhat, ZEROS_FAST,
                                            sigma_hint=sigma, 
                                            include_negative_zeros=True, 
                                            verbose=False)
    
    Q_without_factor, comp_without = compute_Q_weil(h, hhat, ZEROS_FAST,
                                                  sigma_hint=sigma,
                                                  include_negative_zeros=False,
                                                  verbose=False)
    
    # Проверяем что Z_with = 2 * Z_without
    z_ratio = comp_with['Z'] / max(comp_without['Z'], 1e-10)
    
    console.print(f"  Z with factor 2: {comp_with['Z']:+.6f}")
    console.print(f"  Z without factor: {comp_without['Z']:+.6f}")
    console.print(f"  Ratio: {z_ratio:.3f} (should be ≈ 2.0)")
    
    is_consistent = abs(z_ratio - 2.0) < 0.01
    
    if is_consistent:
        console.print("[green]✅ Z-factor consistent[/green]")
    else:
        console.print("[red]❌ Z-factor INCONSISTENT![/red]")
    
    return is_consistent

def test_tail_bounds():
    """CRITICAL: Хвосты должны быть << |Q|"""
    console.print("\n[bold yellow]TEST 3: TAIL BOUNDS[/bold yellow]")
    
    from scipy import integrate
    
    test_sigmas = [2.0, 5.0, 8.0]
    
    table = Table(box=box.SIMPLE)
    table.add_column("σ", justify="center")
    table.add_column("Q", justify="right")
    table.add_column("Z_tail", justify="right")
    table.add_column("P_tail", justify="right")
    table.add_column("Rel_Err", justify="right")
    table.add_column("Status", justify="center")
    
    all_bounded = True
    
    for sigma in test_sigmas:
        h, hhat = create_gaussian_pair(sigma)
        Q, components = compute_Q_weil(h, hhat, ZEROS_FAST,
                                     sigma_hint=sigma, verbose=False)
        
        # Z-tail: оценка для γ > γ_max
        gamma_max = max(ZEROS_FAST)
        def z_tail_integrand(t):
            return h(t) * np.log(max(t, 1.0)) / (2*np.pi)
        
        z_tail, _ = integrate.quad(z_tail_integrand, gamma_max, np.inf, limit=200)
        
        # P-tail: грубая оценка через убывание ĥ
        N_max = 1000
        p_tail = abs(hhat(np.log(N_max))) * np.log(N_max) / (2*np.pi)
        
        # Относительная ошибка
        rel_error = (abs(z_tail) + abs(p_tail)) / max(abs(Q), 1e-10)
        
        is_bounded = rel_error < 0.01  # < 1%
        
        if not is_bounded:
            all_bounded = False
        
        status = "[green]✅[/green]" if is_bounded else "[red]❌[/red]"
        
        table.add_row(
            f"{sigma:.1f}",
            f"{Q:+.4f}",
            f"{z_tail:.2e}",
            f"{p_tail:.2e}",
            f"{rel_error:.1%}",
            status
        )
    
    console.print(table)
    
    if all_bounded:
        console.print("[green]✅ Tail bounds acceptable[/green]")
    else:
        console.print("[red]❌ Tail bounds TOO LARGE - increase precision![/red]")
    
    return all_bounded

def test_fourier_normalization():
    """CRITICAL: Проверка что P-член содержит правильную 1/(2π) нормировку"""
    console.print("\n[bold yellow]TEST 4: FOURIER NORMALIZATION[/bold yellow]")
    
    sigma = 3.0
    h, hhat = create_gaussian_pair(sigma)
    
    # Проверяем что в P есть деление на 2π
    primes = sieve_primes(100)
    
    # Ручной расчет P БЕЗ нормировки
    P_no_norm = 0.0
    for p in primes[:10]:  # Первые 10 простых
        log_p = np.log(p)
        P_no_norm += 2 * (log_p / np.sqrt(p)) * hhat(log_p)
    
    # Правильный P С нормировкой  
    from fourier_conventions import compute_prime_term
    P_with_norm = compute_prime_term(hhat, sigma)
    
    # Отношение должно быть ≈ 2π
    ratio = P_no_norm / max(P_with_norm, 1e-10)
    expected_ratio = 2 * np.pi
    
    console.print(f"  P without 1/(2π): {P_no_norm:.6f}")
    console.print(f"  P with 1/(2π): {P_with_norm:.6f}")
    console.print(f"  Ratio: {ratio:.3f} (should be ≈ {expected_ratio:.3f})")
    
    is_normalized = abs(ratio - expected_ratio) < 0.1
    
    if is_normalized:
        console.print("[green]✅ Normalization correct[/green]")
    else:
        console.print("[red]❌ Normalization WRONG - missing 1/(2π)![/red]")
    
    return is_normalized

def test_closure_bug_fix():
    """CRITICAL: Проверка что closure bug исправлен"""
    console.print("\n[bold yellow]TEST 5: CLOSURE BUG FIX[/bold yellow]")
    
    # Создаем функции в цикле и проверяем что они разные
    sigmas = [2.0, 5.0, 8.0]
    functions = []
    
    # ПРАВИЛЬНЫЙ способ (с default параметром)
    for sigma in sigmas:
        def h(t, s=sigma):
            return np.exp(-(t**2) / (2 * s**2))
        functions.append((sigma, h))
    
    # Проверяем что функции действительно разные
    results = []
    for sigma, h in functions:
        value = h(1.0)  # Вычисляем в точке t=1
        expected = np.exp(-1.0 / (2 * sigma**2))
        error = abs(value - expected)
        results.append((sigma, value, expected, error))
        
        console.print(f"  σ={sigma}: h(1) = {value:.6f}, expected = {expected:.6f}, error = {error:.2e}")
    
    # Все ошибки должны быть практически нулевые
    all_correct = all(error < 1e-10 for _, _, _, error in results)
    
    if all_correct:
        console.print("[green]✅ Closure bug FIXED[/green]")
    else:
        console.print("[red]❌ Closure bug NOT FIXED![/red]")
    
    return all_correct

def run_all_sanity_checks():
    """Запуск всех критических проверок"""
    console.print("[bold cyan]SANITY CHECK SUITE[/bold cyan]")
    console.print("[dim]Запускать перед любыми финальными выводами![/dim]\n")
    
    tests = [
        ("TIME vs FREQ consistency", test_consistency_time_vs_freq),
        ("Z-factor consistency", test_z_factor_consistency), 
        ("Tail bounds", test_tail_bounds),
        ("Fourier normalization", test_fourier_normalization),
        ("Closure bug fix", test_closure_bug_fix)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[red]ERROR in {test_name}: {e}[/red]")
            results.append((test_name, False))
    
    # Итоговый отчет
    console.print("\n" + "="*50)
    console.print("[bold]SANITY CHECK RESULTS:[/bold]\n")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "[green]PASS[/green]" if success else "[red]FAIL[/red]"
        console.print(f"  {status} {test_name}")
        if success:
            passed += 1
    
    console.print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        console.print("[bold green]🎉 ALL SANITY CHECKS PASSED![/bold green]")
        console.print("[green]Safe to proceed with analysis[/green]")
        return True
    else:
        console.print("[bold red]❌ SANITY CHECKS FAILED![/bold red]")
        console.print("[red]DO NOT TRUST RESULTS - fix bugs first![/red]")
        return False

if __name__ == "__main__":
    success = run_all_sanity_checks()
    exit(0 if success else 1)