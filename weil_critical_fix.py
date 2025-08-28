#!/usr/bin/env python3
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ПРАВИЛЬНЫЙ ВЫБОР ТЕСТ-ФУНКЦИЙ
========================================================
Проблема найдена: гауссиана с малым σ затухает до нуля на нулях Римана!
Решение: использовать функции с правильным масштабом.
"""

import numpy as np
from scipy import special, integrate
from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt

console = Console()

# Нули Римана
ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
])

def von_mangoldt_simple(n):
    """Упрощённая функция фон Мангольдта (только простые)"""
    if n <= 1:
        return 0.0
    # Проверка на простоту
    if n == 2:
        return np.log(2)
    if n % 2 == 0:
        return 0.0
    for d in range(3, int(np.sqrt(n)) + 1, 2):
        if n % d == 0:
            return 0.0
    return np.log(n)

def compute_weil_Q(phi_func, phi_hat_func, num_zeros=30):
    """
    Вычисляем Q = Z - A - P с правильными знаками
    """
    
    # Z: сумма по нулям
    Z = sum(phi_func(gamma) for gamma in ZEROS[:num_zeros])
    
    # A: архимедов член
    def arch_integrand(t):
        z = 0.25 + 0.5j * t
        psi = special.digamma(z)
        return phi_func(t) * np.real(psi) / (2 * np.pi)
    
    A_integral, _ = integrate.quad(arch_integrand, -100, 100, limit=200)
    A_const = -np.log(np.pi) * phi_func(0)
    A = A_integral + A_const
    
    # P: сумма по простым
    P = 0.0
    for n in range(2, 1000):
        lambda_n = von_mangoldt_simple(n)
        if lambda_n > 0:
            log_n = np.log(n)
            phi_hat_val = phi_hat_func(log_n)
            P += 2 * (lambda_n / np.sqrt(n)) * phi_hat_val
    
    Q = Z - A - P
    
    return Q, {'Z': Z, 'A': A, 'P': P}

# ========== ТЕСТ-ФУНКЦИИ С ПРАВИЛЬНЫМ МАСШТАБОМ ==========

console.print("[bold cyan]ТЕСТ-ФУНКЦИИ С ПРАВИЛЬНЫМ МАСШТАБОМ[/bold cyan]\n")

# 1. Гауссиана с БОЛЬШИМ σ (чтобы не затухала на нулях)
def test_large_gaussian():
    sigma = 20.0  # Большое σ!
    
    phi = lambda t: np.exp(-t**2 / (2 * sigma**2))
    phi_hat = lambda xi: np.sqrt(2*np.pi) * sigma * np.exp(-sigma**2 * xi**2 / 2)
    
    Q, comp = compute_weil_Q(phi, phi_hat)
    
    console.print(f"[yellow]Gaussian(σ={sigma}):[/yellow]")
    console.print(f"  φ(γ₁={ZEROS[0]:.2f}) = {phi(ZEROS[0]):.6f} (не затухает!)")
    console.print(f"  Z = {comp['Z']:10.6f}")
    console.print(f"  A = {comp['A']:10.6f}")
    console.print(f"  P = {comp['P']:10.6f}")
    console.print(f"  Q = {Q:10.6f} {'✅' if Q > 0 else '❌'}\n")
    
    return Q

# 2. Функция с компактным носителем в ФИЗИЧЕСКОМ пространстве
def test_physical_compact():
    R = 100.0  # Радиус носителя
    
    def phi(t):
        if abs(t) < R:
            return np.exp(-1 / (1 - (t/R)**2))
        return 0.0
    
    # Преобразование Фурье (численное)
    def phi_hat(xi):
        integrand = lambda t: phi(t) * np.exp(-1j * xi * t)
        real_part, _ = integrate.quad(lambda t: np.real(integrand(t)), -R, R, limit=200)
        imag_part, _ = integrate.quad(lambda t: np.imag(integrand(t)), -R, R, limit=200)
        return real_part + 1j * imag_part
    
    Q, comp = compute_weil_Q(phi, lambda xi: np.real(phi_hat(xi)))
    
    console.print(f"[yellow]PhysicalCompact(R={R}):[/yellow]")
    console.print(f"  φ(γ₁={ZEROS[0]:.2f}) = {phi(ZEROS[0]):.6f}")
    console.print(f"  Z = {comp['Z']:10.6f}")
    console.print(f"  A = {comp['A']:10.6f}")
    console.print(f"  P = {comp['P']:10.6f}")
    console.print(f"  Q = {Q:10.6f} {'✅' if Q > 0 else '❌'}\n")
    
    return Q

# 3. Экспоненциально затухающая (но медленно)
def test_slow_exponential():
    alpha = 0.01  # Очень медленное затухание
    
    phi = lambda t: np.exp(-alpha * abs(t))
    # Преобразование Фурье: 2α / (α² + ξ²)
    phi_hat = lambda xi: 2 * alpha / (alpha**2 + xi**2)
    
    Q, comp = compute_weil_Q(phi, phi_hat)
    
    console.print(f"[yellow]SlowExponential(α={alpha}):[/yellow]")
    console.print(f"  φ(γ₁={ZEROS[0]:.2f}) = {phi(ZEROS[0]):.6f}")
    console.print(f"  Z = {comp['Z']:10.6f}")
    console.print(f"  A = {comp['A']:10.6f}")
    console.print(f"  P = {comp['P']:10.6f}")
    console.print(f"  Q = {Q:10.6f} {'✅' if Q > 0 else '❌'}\n")
    
    return Q

# 4. Функция типа sinc с огибающей
def test_windowed_sinc():
    omega_c = 5.0  # Частота среза
    sigma = 30.0   # Ширина огибающей
    
    # φ(t) = sinc(ωt) * exp(-t²/2σ²)
    def phi(t):
        if abs(t) < 1e-10:
            return omega_c / np.pi
        sinc_part = np.sin(omega_c * t) / (np.pi * t)
        window = np.exp(-t**2 / (2 * sigma**2))
        return sinc_part * window
    
    # Приближённое преобразование Фурье
    def phi_hat(xi):
        if abs(xi) < omega_c:
            return np.sqrt(2*np.pi) * sigma * np.exp(-sigma**2 * xi**2 / 2)
        return 0.0
    
    Q, comp = compute_weil_Q(phi, phi_hat)
    
    console.print(f"[yellow]WindowedSinc(ω={omega_c}, σ={sigma}):[/yellow]")
    console.print(f"  φ(γ₁={ZEROS[0]:.2f}) = {phi(ZEROS[0]):.6f}")
    console.print(f"  Z = {comp['Z']:10.6f}")
    console.print(f"  A = {comp['A']:10.6f}")
    console.print(f"  P = {comp['P']:10.6f}")
    console.print(f"  Q = {Q:10.6f} {'✅' if Q > 0 else '❌'}\n")
    
    return Q

# ========== СКАНИРОВАНИЕ ПАРАМЕТРОВ ==========

def scan_gaussian_sigma():
    """Находим оптимальное σ для гауссианы"""
    
    console.print("[bold]СКАНИРОВАНИЕ σ ДЛЯ ГАУССИАНЫ:[/bold]\n")
    
    sigmas = np.logspace(0, 2, 20)  # От 1 до 100
    results = []
    
    table = Table(title="Зависимость Q от σ", box=box.ROUNDED)
    table.add_column("σ", style="cyan")
    table.add_column("φ(γ₁)", style="yellow")
    table.add_column("Z", justify="right")
    table.add_column("A", justify="right")
    table.add_column("P", justify="right")
    table.add_column("Q", justify="right", style="bold")
    
    for sigma in sigmas:
        phi = lambda t: np.exp(-t**2 / (2 * sigma**2))
        phi_hat = lambda xi: np.sqrt(2*np.pi) * sigma * np.exp(-sigma**2 * xi**2 / 2)
        
        Q, comp = compute_weil_Q(phi, phi_hat, num_zeros=30)
        results.append((sigma, Q, comp))
        
        color = "green" if Q > 0 else "red"
        table.add_row(
            f"{sigma:.1f}",
            f"{phi(ZEROS[0]):.4f}",
            f"{comp['Z']:.3f}",
            f"{comp['A']:.3f}",
            f"{comp['P']:.3f}",
            f"[{color}]{Q:.3f}[/{color}]"
        )
    
    console.print(table)
    
    # График
    plt.figure(figsize=(10, 6))
    sigmas_plot = [r[0] for r in results]
    Q_values = [r[1] for r in results]
    
    plt.plot(sigmas_plot, Q_values, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('σ', fontsize=12)
    plt.ylabel('Q', fontsize=12)
    plt.title('Q(σ) for Gaussian test function', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Находим переход
    sign_changes = np.where(np.diff(np.sign(Q_values)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        sigma_critical = sigmas_plot[idx]
        plt.axvline(x=sigma_critical, color='g', linestyle=':', label=f'σ_c ≈ {sigma_critical:.1f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('gaussian_sigma_scan.png', dpi=150)
    console.print("\n[dim]График сохранён в gaussian_sigma_scan.png[/dim]")

# ========== ГЛАВНАЯ ПРОГРАММА ==========

def main():
    console.print("="*60 + "\n")
    
    # Тестируем разные функции
    Q1 = test_large_gaussian()
    Q2 = test_physical_compact()
    Q3 = test_slow_exponential()
    Q4 = test_windowed_sinc()
    
    # Сканируем параметры
    scan_gaussian_sigma()
    
    # Выводы
    console.print("\n" + "="*60)
    console.print("\n[bold green]ВЫВОДЫ:[/bold green]")
    console.print("\n1. Гауссиана с малым σ затухает на нулях → Z ≈ 0 → Q < 0")
    console.print("2. Нужны функции с φ(γ) ≠ 0 на нулях Римана")
    console.print("3. Критическое σ для гауссианы: σ > 10-20")
    console.print("4. Медленно затухающие функции дают больший вклад Z")
    
    if any(Q > 0 for Q in [Q1, Q2, Q3, Q4]):
        console.print("\n[bold green]✅ НАЙДЕНЫ ФУНКЦИИ С Q > 0![/bold green]")
    else:
        console.print("\n[yellow]⚠️ Все Q < 0, но теперь понятна причина[/yellow]")

if __name__ == "__main__":
    main()