#!/usr/bin/env python3
"""
МИНИМАЛЬНЫЙ ТЕСТОВЫЙ СТЕНД ДЛЯ ПРОВЕРКИ WEIL EXPLICIT FORMULA
==============================================================
Единая Фурье-конвенция, кросс-валидация, хвостовые оценки
"""

import numpy as np
from math import log, sqrt, exp, pi
from scipy import special, integrate
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ----- Фурье-конвенция: ĥ(ξ) = ∫ h(t) e^{-iξt} dt
# ----- Парсеваль: ∫|h|^2 dt = (1/2π)∫|ĥ|^2 dξ

def gaussian_pair(sigma):
    """Гауссиана и её преобразование Фурье"""
    def h(t):     
        return np.exp(-(t**2)/(2*sigma**2))
    def hhat(xi): 
        return sqrt(2*pi)*sigma*np.exp(-(sigma**2)*(xi**2)/2)
    return h, hhat

# ----- Первые 30 нулей Римана
ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
]

# ----- Простые до 600
def sieve_of_eratosthenes(n):
    """Решето Эратосфена для простых до n"""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = sieve_of_eratosthenes(600)

# ----- Архимедов член (в единой нормировке)
def A_term(h, sigma, tmax=None, eps=1e-10):
    """Архимедов член с корректным весом"""
    if tmax is None: 
        tmax = max(10*sigma, 50.)
    def integrand(t):
        z   = 0.25 + 0.5j*t
        psi = special.digamma(z)
        return h(t)*(np.real(psi)-np.log(pi))/(2*pi)
    val, err = integrate.quad(integrand, -tmax, tmax, epsrel=eps, limit=400)
    return val, err

# ----- Простой член с адаптивным срезом и 1/(2π)
def P_term(hhat, sigma, primes=None, tol=1e-12):
    """Простой член с правильной нормировкой"""
    if primes is None:
        primes = PRIMES
        
    # Адаптивный срез по хвосту гауссианы
    L = np.sqrt(2.0*np.log((np.sqrt(2*pi)*sigma)/tol))/max(sigma,1e-12)
    Nmx = min(int(np.exp(L))+100, 10000)
    
    P = 0.0
    tail_count = 0
    for p in primes:
        if p>Nmx: 
            tail_count = len([x for x in primes if x > Nmx])
            break
        lp = log(p)
        # Основная формула с 1/(2π)
        P += (2*(lp/sqrt(p))*hhat(lp))/(2*pi)
        # Степени простого
        pk=p*p
        while pk<=Nmx:
            P += (2*(lp/sqrt(pk))*hhat(log(pk)))/(2*pi)
            pk*=p
    return P, tail_count

# ----- Сумма по нулям
def Z_term(h, zeros=None):
    """Сумма по нулям Римана"""
    if zeros is None:
        zeros = ZEROS
    return sum(h(float(g)) for g in zeros)

# ----- Q(h) = Z - A - P
def Q_of_h(sigma, zeros=None, primes=None, verbose=True):
    """Полное вычисление Q для гауссианы"""
    h, hhat = gaussian_pair(sigma)
    
    Z = Z_term(h, zeros)
    A, A_err = A_term(h, sigma)
    P, P_tail = P_term(hhat, sigma, primes)
    
    Q = Z - A - P
    
    if verbose:
        console.print(f"\n[cyan]σ = {sigma:.3f}:[/cyan]")
        console.print(f"  Z = {Z:+.6f} (сумма по {len(zeros if zeros else ZEROS)} нулям)")
        console.print(f"  A = {A:+.6f} (ошибка ≈ {A_err:.2e})")
        console.print(f"  P = {P:+.6f} (пропущено {P_tail} простых в хвосте)")
        console.print(f"  [bold]Q = {Q:+.6f}[/bold] {'✅' if Q > 0 else '❌'}")
    
    return Q, {'Z': Z, 'A': A, 'P': P, 'A_err': A_err, 'P_tail': P_tail}

# ----- Поляризация: Q(h1,h2) = 0.5*(Q(h1+h2)-Q(h1)-Q(h2))
def Q_bilinear(sig1, sig2, zeros=None, primes=None):
    """Билинейная форма Q через поляризацию"""
    h1, hhat1 = gaussian_pair(sig1)
    h2, hhat2 = gaussian_pair(sig2)
    
    # Сумма функций
    def hsum(t): 
        return h1(t) + h2(t)
    def hhat_sum(xi):
        return hhat1(xi) + hhat2(xi)
    
    # Вычисляем три значения Q
    Q1, _ = Q_of_h(sig1, zeros, primes, verbose=False)
    Q2, _ = Q_of_h(sig2, zeros, primes, verbose=False)
    
    # Для суммы нужен средний параметр для архимедова члена
    sig_avg = (sig1 + sig2) / 2
    h_avg, _ = gaussian_pair(sig_avg)
    
    Z12 = Z_term(hsum, zeros)
    A12, _ = A_term(hsum, sig_avg)
    P12, _ = P_term(hhat_sum, sig_avg, primes)
    Q12 = Z12 - A12 - P12
    
    return 0.5*(Q12 - Q1 - Q2)

# ----- Хвостовые оценки
def tail_bounds_Z(sigma, Gamma_max):
    """Оценка хвоста Z-суммы"""
    # Z-хвост ~ ∫_{Γ}^∞ h(t) * плотность нулей
    # Грубо: плотность ~ log(t)/(2π)
    f = lambda t: np.exp(-(t*t)/(2*sigma*sigma))*np.log(max(t,1.0))/(2*pi)
    val, _ = integrate.quad(f, Gamma_max, np.inf, limit=200)
    return val

def tail_bounds_P(sigma, N_max):
    """Оценка хвоста P-суммы"""
    # P-хвост ~ Σ_{n>N} Λ(n)/√n * ĥ(log n)/(2π)
    # Грубая оценка через плотность простых
    _, hhat = gaussian_pair(sigma)
    # Используем приближение: π(x) ~ x/log(x)
    tail_est = 0
    for log_n in np.linspace(np.log(N_max), np.log(N_max*10), 50):
        n = np.exp(log_n)
        density = 1/log_n  # Плотность простых
        tail_est += (log_n / sqrt(n)) * hhat(log_n) * density / (2*pi)
    return tail_est * np.log(10)  # Масштабируем на интервал

# ==================== ОСНОВНОЙ КОД ====================

if __name__ == "__main__":
    console.print("[bold cyan]ТЕСТОВЫЙ СТЕНД ДЛЯ КРИТЕРИЯ ВЕЙЛЯ[/bold cyan]")
    console.print("[dim]Единая Фурье-конвенция с правильной нормировкой[/dim]\n")
    
    # 1. Проверка критических значений
    console.print("[yellow]1. ПРОВЕРКА КРИТИЧЕСКИХ ЗНАЧЕНИЙ σ:[/yellow]")
    
    test_sigmas = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    results = []
    
    table = Table(box=box.ROUNDED)
    table.add_column("σ", style="cyan", justify="center")
    table.add_column("Z", justify="right")
    table.add_column("A", justify="right")
    table.add_column("P", justify="right")
    table.add_column("Q", justify="right", style="bold")
    table.add_column("Status", justify="center")
    
    for sigma in test_sigmas:
        Q, comp = Q_of_h(sigma, verbose=False)
        results.append((sigma, Q, comp))
        
        status = "[green]✅[/green]" if Q > 0 else "[red]❌[/red]"
        q_color = "green" if Q > 0 else "red"
        
        table.add_row(
            f"{sigma:.1f}",
            f"{comp['Z']:+.4f}",
            f"{comp['A']:+.4f}",
            f"{comp['P']:+.4f}",
            f"[{q_color}]{Q:+.4f}[/{q_color}]",
            status
        )
    
    console.print(table)
    
    # 2. Поиск переходов через 0
    console.print("\n[yellow]2. ПЕРЕХОДЫ ЧЕРЕЗ НОЛЬ:[/yellow]")
    
    transitions = []
    for i in range(len(results) - 1):
        if results[i][1] * results[i+1][1] < 0:
            s1, s2 = results[i][0], results[i+1][0]
            transitions.append((s1, s2))
            
            # Бинарный поиск точного перехода
            left, right = s1, s2
            while right - left > 0.001:
                mid = (left + right) / 2
                Q_mid, _ = Q_of_h(mid, verbose=False)
                Q_left, _ = Q_of_h(left, verbose=False)
                if Q_left * Q_mid < 0:
                    right = mid
                else:
                    left = mid
            
            sigma_critical = (left + right) / 2
            console.print(f"  Переход при σ ≈ {sigma_critical:.3f}")
            
            # Характер перехода
            Q_before, _ = Q_of_h(sigma_critical - 0.01, verbose=False)
            Q_after, _ = Q_of_h(sigma_critical + 0.01, verbose=False)
            if Q_before < 0 and Q_after > 0:
                console.print(f"    Характер: - → + (Q становится положительным)")
            elif Q_before > 0 and Q_after < 0:
                console.print(f"    Характер: + → - (Q становится отрицательным)")
    
    if not transitions:
        console.print("[green]  Переходов не найдено! Q сохраняет знак.[/green]")
    
    # 3. Грам-матрица для проверки PSD
    console.print("\n[yellow]3. ГРАМ-МАТРИЦА (проверка PSD):[/yellow]")
    
    gram_sigmas = [2.0, 3.0, 4.0, 5.0, 6.0]
    n = len(gram_sigmas)
    G = np.zeros((n, n))
    
    console.print(f"  Вычисление {n}×{n} матрицы для σ = {gram_sigmas}...")
    
    for i, sig1 in enumerate(gram_sigmas):
        for j, sig2 in enumerate(gram_sigmas):
            if j >= i:  # Используем симметрию
                G[i, j] = Q_bilinear(sig1, sig2)
                G[j, i] = G[i, j]
    
    # Собственные значения
    eigenvalues = np.linalg.eigvals(G)
    eigenvalues = np.sort(eigenvalues)
    
    console.print("\n  Собственные значения Грам-матрицы:")
    for i, lam in enumerate(eigenvalues):
        color = "green" if lam >= -1e-10 else "red"
        console.print(f"    λ_{i+1} = [{color}]{lam:+.6f}[/{color}]")
    
    min_eigenvalue = np.min(eigenvalues)
    if min_eigenvalue >= -1e-10:
        console.print(f"\n  [bold green]✅ Матрица PSD! (min λ = {min_eigenvalue:.2e})[/bold green]")
    else:
        console.print(f"\n  [bold red]❌ Матрица НЕ PSD! (min λ = {min_eigenvalue:.2e})[/bold red]")
    
    # 4. Хвостовые оценки
    console.print("\n[yellow]4. ХВОСТОВЫЕ ОЦЕНКИ:[/yellow]")
    
    sigma_test = 5.0
    Q_full, comp = Q_of_h(sigma_test, verbose=False)
    
    Z_tail = tail_bounds_Z(sigma_test, ZEROS[-1])
    P_tail = tail_bounds_P(sigma_test, 600)
    
    console.print(f"\n  Для σ = {sigma_test}:")
    console.print(f"    Z-хвост (γ > {ZEROS[-1]:.1f}): ≈ {Z_tail:.2e}")
    console.print(f"    P-хвост (n > 600): ≈ {P_tail:.2e}")
    console.print(f"    A-ошибка квадратуры: ≈ {comp['A_err']:.2e}")
    console.print(f"\n    Полное Q = {Q_full:+.6f}")
    console.print(f"    Относительная ошибка: < {(abs(Z_tail) + abs(P_tail) + comp['A_err'])/max(abs(Q_full), 1e-10):.1%}")
    
    # 5. Финальный вывод
    console.print("\n" + "="*60)
    console.print("[bold green]ФИНАЛЬНЫЕ ВЫВОДЫ:[/bold green]\n")
    
    positive_range = []
    current_start = None
    
    for sigma, Q, _ in results:
        if Q > 0:
            if current_start is None:
                current_start = sigma
        else:
            if current_start is not None:
                positive_range.append((current_start, sigma))
                current_start = None
    
    if current_start is not None:
        positive_range.append((current_start, results[-1][0]))
    
    if positive_range:
        console.print("[bold]Интервалы где Q > 0:[/bold]")
        for start, end in positive_range:
            console.print(f"  σ ∈ [{start:.1f}, {end:.1f}]")
    else:
        console.print("[red]Q < 0 для всех протестированных σ![/red]")
    
    console.print("\n[cyan]Критерий Вейля требует Q ≥ 0 для ВСЕХ допустимых тест-функций.[/cyan]")
    console.print("[dim]Положительность для одной параметрической семьи согласуется с RH, но не доказывает её.[/dim]")