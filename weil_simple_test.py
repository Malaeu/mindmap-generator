#!/usr/bin/env python3
"""
ПРОСТОЙ ТЕСТ: ТОЛЬКО ГАУССИАНА БЕЗ СЛОЖНОСТЕЙ
=============================================
Проверяем базовую формулу без всяких окон и сплайнов
"""

import numpy as np
from scipy import special
from rich.console import Console

console = Console()

# Нули Римана
ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
])

# Простые числа
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

def test_gaussian(sigma=10.0):
    """Простой тест с гауссианой"""
    
    console.print(f"\n[bold cyan]ТЕСТ ГАУССИАНЫ σ={sigma}[/bold cyan]\n")
    
    # Определяем функции
    def h(t):
        return np.exp(-t**2 / (2 * sigma**2))
    
    def h_hat(xi):
        return np.sqrt(2 * np.pi) * sigma * np.exp(-sigma**2 * xi**2 / 2)
    
    # 1. Z-член - ПРАВИЛЬНО: используем h(γ), НЕ ĥ(γ)!
    # Формула: Z = Σ_ρ h(Im ρ)
    Z = 0.0
    console.print("[yellow]Z-член (сумма по нулям):[/yellow]")
    for i, gamma in enumerate(ZEROS):
        contrib = h(gamma)
        Z += contrib
        if i < 3:
            console.print(f"  γ={gamma:.2f}: h(γ)={contrib:.6e}")
    console.print(f"  [bold]Z = {Z:.6f}[/bold]\n")
    
    # 2. A-член - архимедов
    # A = φ(0)·(-log π) + интеграл
    from scipy import integrate
    
    def arch_integrand(t):
        z = 0.25 + 0.5j * t
        psi = special.digamma(z)
        weight = np.real(psi)
        return h(t) * weight / (2 * np.pi)
    
    A_integral, _ = integrate.quad(arch_integrand, -50, 50, limit=200)
    A_const = -np.log(np.pi) * h(0)
    A = A_integral + A_const
    
    console.print("[yellow]A-член (архимедов):[/yellow]")
    console.print(f"  Интеграл: {A_integral:.6f}")
    console.print(f"  -log(π)·h(0): {A_const:.6f}")
    console.print(f"  [bold]A = {A:.6f}[/bold]\n")
    
    # 3. P-член - простые
    # P = 2·Σ (log p / √p)·ĥ(log p)
    P = 0.0
    console.print("[yellow]P-член (простые):[/yellow]")
    for p in PRIMES[:5]:
        log_p = np.log(p)
        contrib = 2 * (log_p / np.sqrt(p)) * h_hat(log_p)
        P += contrib
        console.print(f"  p={p}: 2·Λ(p)/√p·ĥ(log p) = {contrib:.6f}")
    
    # Добавляем остальные простые
    for p in PRIMES[5:]:
        log_p = np.log(p)
        P += 2 * (log_p / np.sqrt(p)) * h_hat(log_p)
    
    console.print(f"  [bold]P = {P:.6f}[/bold]\n")
    
    # Квадратичная форма
    Q = Z - A - P
    
    console.print("[bold]РЕЗУЛЬТАТ:[/bold]")
    console.print(f"  Z = {Z:.6f}")
    console.print(f"  A = {A:.6f}")
    console.print(f"  P = {P:.6f}")
    console.print(f"  [bold]Q = Z - A - P = {Q:.6f}[/bold]")
    
    if Q > 0:
        console.print("  [bold green]✅ Q > 0 - критерий выполнен![/bold green]")
    else:
        console.print("  [bold red]❌ Q < 0[/bold red]")
    
    return Q

# Тестируем разные σ
console.print("[bold]СКАНИРОВАНИЕ σ[/bold]")

sigmas = [1, 2, 5, 10, 20, 50, 100]
results = []

for sigma in sigmas:
    Q = test_gaussian(sigma)
    results.append((sigma, Q))
    console.print("-" * 40)

# Итоговая таблица
console.print("\n[bold]СВОДКА:[/bold]")
for sigma, Q in results:
    status = "✅" if Q > 0 else "❌"
    console.print(f"σ={sigma:3}: Q={Q:10.6f} {status}")

# Находим переход
positive = [s for s, q in results if q > 0]
if positive:
    console.print(f"\n[bold green]Критическое σ ≈ {min(positive)}[/bold green]")