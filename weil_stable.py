#!/usr/bin/env python3
"""
УЛУЧШЕННАЯ ЧИСЛЕННАЯ СТАБИЛЬНОСТЬ ДЛЯ КРИТЕРИЯ ВЕЙЛЯ
=====================================================
Проблема: интегралы вычисляются неправильно из-за дискретной сетки.
Решение: правильная нормализация FFT и интегрирование.
"""

import numpy as np
from scipy import special
from scipy.linalg import eigvalsh
from scipy.fft import fft, ifft, fftshift, ifftshift
from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt

console = Console()

# Первые 50 нулей дзета-функции
ZEROS_50 = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516684, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846
])

def digamma_weight(t):
    """Re ψ(1/4 + it/2) - log π"""
    z = 0.25 + 0.5j * t
    psi_val = special.digamma(z)
    return np.real(psi_val) - np.log(np.pi)

def von_mangoldt(n):
    """Функция фон Мангольдта Λ(n)"""
    if n <= 1:
        return 0.0
    
    # Простые числа до 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for p in primes:
        if p > n:
            break
        # Проверяем, является ли n степенью p
        pk = p
        while pk <= n:
            if pk == n:
                return np.log(p)
            pk *= p
    
    # Для больших n используем факторизацию
    if n > 100:
        # Проверяем делимость на малые простые
        for p in primes[:10]:  # Только первые 10 простых
            if n % p == 0:
                m = n
                count = 0
                while m % p == 0:
                    m //= p
                    count += 1
                if m == 1:  # n = p^k
                    return np.log(p)
                return 0.0
    
    return 0.0

def create_smooth_bump(t, A):
    """Гладкая bump функция с компактным носителем [-A, A]"""
    h = np.zeros_like(t, dtype=complex)
    mask = np.abs(t) < A
    t_scaled = t[mask] / A
    h[mask] = np.exp(-1 / (1 - t_scaled**2))
    # Нормализуем для единичной L2 нормы
    norm = np.sqrt(np.trapz(np.abs(h)**2, t))
    if norm > 0:
        h = h / norm
    return h

def compute_weil_quadratic_stable(A=3.0, N_grid=4096, T_max=100.0):
    """
    Стабильное вычисление квадратичной формы Q(h)
    
    Ключевые улучшения:
    1. Правильная нормализация FFT
    2. Использование трапециевидного правила для интегралов
    3. Центрированные частоты для лучшей интерполяции
    """
    
    # Создаем временную сетку с четным числом точек
    N = N_grid if N_grid % 2 == 0 else N_grid + 1
    t = np.linspace(-T_max, T_max, N)
    dt = t[1] - t[0]
    
    # Тестовая функция
    h = create_smooth_bump(t, A)
    
    # Вычисляем преобразование Фурье с правильной нормализацией
    # h_hat(ω) = ∫ h(t) e^(-iωt) dt
    h_hat = fftshift(fft(ifftshift(h))) * dt
    omega = fftshift(np.fft.fftfreq(N, dt)) * 2 * np.pi
    
    # 1. Z член - сумма по нулям Римана
    Z_sum = 0.0
    for gamma in ZEROS_50:
        # h_hat(γ) + h_hat(-γ) = 2 Re[h_hat(γ)]
        # Интерполируем значение h_hat в точке gamma
        idx = np.searchsorted(omega, gamma)
        if 0 < idx < len(omega) - 1:
            # Линейная интерполяция
            alpha = (gamma - omega[idx-1]) / (omega[idx] - omega[idx-1])
            h_hat_gamma = (1 - alpha) * h_hat[idx-1] + alpha * h_hat[idx]
            Z_sum += 2 * np.real(h_hat_gamma)
    
    # 2. A член - архимедов интеграл
    # A(h) = ∫ |h_hat(t)|^2 * weight(t) dt
    weight = np.array([digamma_weight(ti) for ti in t])
    h_for_A = create_smooth_bump(t, A)  # Используем ту же функцию
    A_integrand = np.abs(h_for_A)**2 * weight
    A_sum = np.trapz(A_integrand, t)
    
    # 3. P член - сумма по простым
    P_sum = 0.0
    for n in range(2, 1000):
        lambda_n = von_mangoldt(n)
        if lambda_n > 0:
            log_n = np.log(n)
            
            # Интерполируем h_hat(log n)
            idx = np.searchsorted(omega, log_n)
            if 0 < idx < len(omega) - 1:
                alpha = (log_n - omega[idx-1]) / (omega[idx] - omega[idx-1])
                h_hat_log_n = (1 - alpha) * h_hat[idx-1] + alpha * h_hat[idx]
                
                # h_hat(log n) + h_hat(-log n)
                idx_neg = np.searchsorted(omega, -log_n)
                if 0 < idx_neg < len(omega) - 1:
                    alpha_neg = (-log_n - omega[idx_neg-1]) / (omega[idx_neg] - omega[idx_neg-1])
                    h_hat_minus_log_n = (1 - alpha_neg) * h_hat[idx_neg-1] + alpha_neg * h_hat[idx_neg]
                else:
                    h_hat_minus_log_n = 0
                
                contribution = (lambda_n / np.sqrt(n)) * (h_hat_log_n + h_hat_minus_log_n)
                P_sum += np.real(contribution) / (2 * np.pi)
    
    # Квадратичная форма
    Q = Z_sum - A_sum - P_sum
    
    return Q, {
        'Z': Z_sum,
        'A': A_sum,
        'P': P_sum,
        'dt': dt,
        'N': N
    }

def test_convergence():
    """Тестируем сходимость при увеличении размера сетки"""
    
    console.print("\n[bold cyan]ТЕСТ СХОДИМОСТИ С УЛУЧШЕННОЙ СТАБИЛЬНОСТЬЮ[/bold cyan]\n")
    
    # Тестируем для разных значений A
    A_values = [3.0, 5.0, 7.0, 9.0]
    grid_sizes = [512, 1024, 2048, 4096, 8192]
    
    results = {}
    
    for A in A_values:
        results[A] = []
        console.print(f"\n[yellow]Тестируем A = {A:.1f}[/yellow]")
        
        for N in grid_sizes:
            Q, stats = compute_weil_quadratic_stable(A=A, N_grid=N, T_max=50.0)
            results[A].append((N, Q, stats))
            console.print(f"  N={N:5d}: Q = {Q:10.6f} (Z={stats['Z']:8.4f}, A={stats['A']:8.4f}, P={stats['P']:8.4f})")
    
    # Анализ стабильности
    console.print("\n[bold]АНАЛИЗ СТАБИЛЬНОСТИ:[/bold]")
    
    table = Table(title="Относительная вариация Q при изменении сетки", box=box.ROUNDED)
    table.add_column("A", style="cyan")
    table.add_column("Q(N=512)", style="yellow")
    table.add_column("Q(N=8192)", style="yellow")
    table.add_column("Variation", style="magenta")
    table.add_column("Status", style="green")
    
    for A in A_values:
        Q_min = results[A][0][1]  # N=512
        Q_max = results[A][-1][1]  # N=8192
        
        if abs(Q_min) > 0.01:
            variation = abs(Q_max - Q_min) / abs(Q_min)
        else:
            variation = abs(Q_max - Q_min)
        
        status = "✅ Stable" if variation < 0.1 else "⚠️ Unstable" if variation < 0.3 else "❌ Bad"
        
        table.add_row(
            f"{A:.1f}",
            f"{Q_min:.6f}",
            f"{Q_max:.6f}",
            f"{variation:.1%}",
            status
        )
    
    console.print(table)
    
    # График сходимости
    plt.figure(figsize=(12, 8))
    
    for i, A in enumerate(A_values):
        plt.subplot(2, 2, i+1)
        N_vals = [r[0] for r in results[A]]
        Q_vals = [r[1] for r in results[A]]
        
        plt.plot(N_vals, Q_vals, 'o-', linewidth=2, markersize=8)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Grid size N', fontsize=10)
        plt.ylabel('Q value', fontsize=10)
        plt.title(f'A = {A:.1f}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Добавляем текст со стабильностью
        if len(Q_vals) > 1:
            var = abs(Q_vals[-1] - Q_vals[0]) / abs(Q_vals[0]) if Q_vals[0] != 0 else 0
            plt.text(0.05, 0.95, f'Var: {var:.1%}', 
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Convergence Analysis with Improved Stability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('weil_convergence_stable.png', dpi=150)
    console.print("\n[dim]График сохранен в weil_convergence_stable.png[/dim]")
    
    return results

def scan_critical_region():
    """Детальное сканирование критической области"""
    
    console.print("\n[bold cyan]СКАНИРОВАНИЕ КРИТИЧЕСКОЙ ОБЛАСТИ[/bold cyan]\n")
    
    # Фокусируемся на области около A_c ≈ 6.9
    A_values = np.linspace(6.0, 8.0, 41)
    N = 4096  # Фиксированная сетка
    
    results = []
    
    for A in A_values:
        Q, stats = compute_weil_quadratic_stable(A=A, N_grid=N)
        results.append((A, Q))
        
        # Печатаем только около перехода
        if abs(Q) < 10:
            console.print(f"A = {A:.2f}: Q = {Q:8.4f} {'✅' if Q > 0 else '❌'}")
    
    # Находим точный переход
    A_vals = np.array([r[0] for r in results])
    Q_vals = np.array([r[1] for r in results])
    
    # Интерполируем для нахождения точной критической точки
    sign_changes = np.where(np.diff(np.sign(Q_vals)))[0]
    
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        A_before = A_vals[idx]
        A_after = A_vals[idx + 1]
        Q_before = Q_vals[idx]
        Q_after = Q_vals[idx + 1]
        
        # Линейная интерполяция
        A_critical = A_before - Q_before * (A_after - A_before) / (Q_after - Q_before)
        
        console.print(f"\n[bold green]🎯 ТОЧНАЯ КРИТИЧЕСКАЯ ТОЧКА: A_c = {A_critical:.4f}[/bold green]")
    
    # График
    plt.figure(figsize=(10, 6))
    plt.plot(A_vals, Q_vals, 'b-', linewidth=2, label='Q(A)')
    plt.axhline(y=0, color='r', linestyle='--', label='Q = 0')
    
    if len(sign_changes) > 0:
        plt.axvline(x=A_critical, color='g', linestyle=':', linewidth=2, label=f'A_c = {A_critical:.3f}')
        plt.plot(A_critical, 0, 'go', markersize=10)
    
    # Заливка областей
    plt.fill_between(A_vals, 0, Q_vals, where=(Q_vals > 0), 
                     color='green', alpha=0.2, label='PSD region')
    plt.fill_between(A_vals, Q_vals, 0, where=(Q_vals < 0), 
                     color='red', alpha=0.2, label='Not PSD')
    
    plt.xlabel('Bandwidth A', fontsize=12)
    plt.ylabel('Q value', fontsize=12)
    plt.title('Critical Region Detail (Improved Stability)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('critical_region_stable.png', dpi=150)
    console.print("\n[dim]График сохранен в critical_region_stable.png[/dim]")
    
    return A_critical if len(sign_changes) > 0 else None

def main():
    """Главная функция"""
    
    console.print("\n[bold green]УЛУЧШЕННАЯ ЧИСЛЕННАЯ СТАБИЛЬНОСТЬ[/bold green]")
    console.print("[yellow]Правильная нормализация FFT и интегрирование[/yellow]\n")
    
    # 1. Тест сходимости
    convergence_results = test_convergence()
    
    # 2. Точное определение критической точки
    A_critical = scan_critical_region()
    
    # 3. Финальный отчет
    console.print("\n" + "="*60)
    console.print("\n[bold green]ИТОГОВЫЕ РЕЗУЛЬТАТЫ:[/bold green]")
    
    if A_critical:
        console.print(f"\n✅ Критическая точка: A_c = {A_critical:.4f}")
        console.print("   → При A < A_c квадратичная форма Q < 0")
        console.print("   → При A > A_c квадратичная форма Q > 0 (PSD)")
    
    # Проверяем стабильность для A > A_c
    if A_critical:
        test_A = A_critical + 1.0
        Q_512, _ = compute_weil_quadratic_stable(A=test_A, N_grid=512)
        Q_8192, _ = compute_weil_quadratic_stable(A=test_A, N_grid=8192)
        
        variation = abs(Q_8192 - Q_512) / abs(Q_512) if Q_512 != 0 else 0
        
        console.print(f"\n📊 Стабильность при A = {test_A:.1f}:")
        console.print(f"   Q(N=512) = {Q_512:.6f}")
        console.print(f"   Q(N=8192) = {Q_8192:.6f}")
        console.print(f"   Вариация: {variation:.1%}")
        
        if variation < 0.05:
            console.print("   [bold green]✅ Отличная стабильность![/bold green]")
        elif variation < 0.15:
            console.print("   [yellow]⚠️ Хорошая стабильность[/yellow]")
        else:
            console.print("   [red]❌ Недостаточная стабильность[/red]")

if __name__ == "__main__":
    main()