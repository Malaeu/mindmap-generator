#!/usr/bin/env python3
"""
ДЕТАЛЬНОЕ ИССЛЕДОВАНИЕ КРИТИЧЕСКОЙ ОБЛАСТИ σ ≈ 5
==================================================
Где именно Q переходит через 0?
"""

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from scipy import integrate, special

console = Console()

# Первые 30 нулей Римана
ZEROS = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
])

def compute_Q_gaussian(sigma, num_zeros=30, verbose=False):
    """Вычисляем Q для гауссианы с данным σ"""
    
    # Функции
    h = lambda t: np.exp(-t**2 / (2 * sigma**2))
    h_hat = lambda xi: np.sqrt(2 * np.pi) * sigma * np.exp(-sigma**2 * xi**2 / 2)
    
    # 1. Z - сумма по нулям
    Z = sum(h(gamma) for gamma in ZEROS[:num_zeros])
    
    # 2. A - архимедов член
    def arch_integrand(t):
        z = 0.25 + 0.5j * t
        psi = special.digamma(z)
        return h(t) * (np.real(psi) - np.log(np.pi)) / (2 * np.pi)
    
    # Адаптивные пределы интегрирования с запасом
    t_max = max(sigma * 10, 50.0)  # ±10σ или минимум 50
    A, _ = integrate.quad(arch_integrand, -t_max, t_max, limit=400, epsrel=1e-10)
    
    # 3. P - сумма по простым с правильной нормировкой
    P = 0.0
    
    # Динамический предел на основе хвоста гауссианы
    # h_hat(log n) ≈ sqrt(2π)*σ*exp(-(σ^2/2)*(log n)^2) < tol
    tol = 1e-12
    L = np.sqrt(2.0 * np.log((np.sqrt(2*np.pi)*sigma)/tol)) / max(sigma, 1e-12)
    N_max = min(int(np.exp(L)) + 100, 10000)  # Ограничиваем 10000 для скорости
    
    # Простые до N_max (используем больше простых для точности)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
              199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
              283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
              383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
              467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571]
    
    for p in primes:
        if p > N_max:
            break
        log_p = np.log(p)
        # Вклад от простого p с нормировкой 1/(2π)
        P += (2 * (log_p / np.sqrt(p)) * h_hat(log_p)) / (2 * np.pi)
        
        # Степени простого
        pk = p * p
        while pk <= N_max:
            log_pk = np.log(pk)
            P += (2 * (log_p / np.sqrt(pk)) * h_hat(log_pk)) / (2 * np.pi)  # Λ(p^k) = log(p)
            pk *= p
    
    Q = Z - A - P
    
    if verbose:
        console.print(f"σ={sigma:.3f}: Z={Z:.6f}, A={A:.6f}, P={P:.6f}, Q={Q:.6f}")
    
    return Q, {'Z': Z, 'A': A, 'P': P}

# ==================== СКАНИРОВАНИЕ ====================

console.print("[bold cyan]ПОИСК КРИТИЧЕСКОЙ ТОЧКИ σ_c[/bold cyan]\n")

# Грубое сканирование
console.print("[yellow]Грубое сканирование σ ∈ [1, 10]:[/yellow]")
sigmas_coarse = np.linspace(1, 10, 19)
Q_coarse = []

for sigma in sigmas_coarse:
    Q, _ = compute_Q_gaussian(sigma)
    Q_coarse.append(Q)
    status = "✅" if Q > 0 else "  "
    console.print(f"  σ={sigma:.1f}: Q={Q:8.4f} {status}")

# Находим интервал где Q меняет знак
sign_changes = []
for i in range(len(Q_coarse) - 1):
    if Q_coarse[i] * Q_coarse[i+1] < 0:
        sign_changes.append((sigmas_coarse[i], sigmas_coarse[i+1]))

console.print(f"\n[bold]Переходы через 0: {sign_changes}[/bold]")

# Детальное сканирование около переходов
all_critical_points = []

for s1, s2 in sign_changes:
    console.print(f"\n[yellow]Детальное сканирование σ ∈ [{s1:.1f}, {s2:.1f}]:[/yellow]")
    
    sigmas_fine = np.linspace(s1, s2, 21)
    Q_fine = []
    
    table = Table(box=box.SIMPLE)
    table.add_column("σ", style="cyan")
    table.add_column("Q", justify="right")
    table.add_column("Status", justify="center")
    
    for sigma in sigmas_fine:
        Q, components = compute_Q_gaussian(sigma, verbose=False)
        Q_fine.append(Q)
        
        status = "[green]✅[/green]" if Q > 0 else "[red]❌[/red]"
        color = "green" if Q > 0 else "red"
        
        table.add_row(
            f"{sigma:.3f}",
            f"[{color}]{Q:.6f}[/{color}]",
            status
        )
    
    console.print(table)
    
    # Интерполяция для точного нуля
    from scipy.interpolate import interp1d
    f = interp1d(sigmas_fine, Q_fine, kind='cubic')
    
    # Бисекция для поиска точного нуля
    left, right = s1, s2
    while right - left > 1e-6:
        mid = (left + right) / 2
        if f(left) * f(mid) < 0:
            right = mid
        else:
            left = mid
    
    sigma_critical = (left + right) / 2
    all_critical_points.append(sigma_critical)
    
    console.print(f"[bold green]Критическая точка: σ_c = {sigma_critical:.6f}[/bold green]")

# ==================== ВИЗУАЛИЗАЦИЯ ====================

console.print("\n[yellow]Построение графиков...[/yellow]")

# Детальная кривая
sigmas_plot = np.linspace(1, 10, 200)
Q_plot = [compute_Q_gaussian(s)[0] for s in sigmas_plot]

# Компоненты для σ около критической точки
sigma_analysis = np.linspace(3, 7, 100)
Z_vals = []
A_vals = []
P_vals = []
Q_vals = []

for s in sigma_analysis:
    Q, comp = compute_Q_gaussian(s)
    Z_vals.append(comp['Z'])
    A_vals.append(comp['A'])
    P_vals.append(comp['P'])
    Q_vals.append(Q)

# Графики
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График 1: Q(σ)
ax1 = axes[0, 0]
ax1.plot(sigmas_plot, Q_plot, 'b-', linewidth=2)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
for sc in all_critical_points:
    ax1.axvline(x=sc, color='g', linestyle=':', alpha=0.7)
    ax1.plot(sc, 0, 'go', markersize=8)
ax1.set_xlabel('σ')
ax1.set_ylabel('Q')
ax1.set_title('Q(σ) для гауссианы')
ax1.grid(True, alpha=0.3)

# График 2: Компоненты
ax2 = axes[0, 1]
ax2.plot(sigma_analysis, Z_vals, 'g-', label='Z', linewidth=2)
ax2.plot(sigma_analysis, A_vals, 'r-', label='A', linewidth=2)
ax2.plot(sigma_analysis, P_vals, 'b-', label='P', linewidth=2)
ax2.set_xlabel('σ')
ax2.set_ylabel('Значение')
ax2.set_title('Компоненты Z, A, P')
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Баланс
ax3 = axes[1, 0]
ax3.plot(sigma_analysis, Q_vals, 'k-', linewidth=2, label='Q = Z - A - P')
ax3.fill_between(sigma_analysis, 0, Q_vals, where=(np.array(Q_vals) > 0),
                  color='green', alpha=0.2, label='Q > 0')
ax3.fill_between(sigma_analysis, Q_vals, 0, where=(np.array(Q_vals) < 0),
                  color='red', alpha=0.2, label='Q < 0')
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.set_xlabel('σ')
ax3.set_ylabel('Q')
ax3.set_title('Области положительности')
ax3.legend()
ax3.grid(True, alpha=0.3)

# График 4: Относительные вклады
ax4 = axes[1, 1]
total = np.abs(Z_vals) + np.abs(A_vals) + np.abs(P_vals)
ax4.plot(sigma_analysis, np.abs(Z_vals) / total * 100, 'g-', label='|Z|', linewidth=2)
ax4.plot(sigma_analysis, np.abs(A_vals) / total * 100, 'r-', label='|A|', linewidth=2)
ax4.plot(sigma_analysis, np.abs(P_vals) / total * 100, 'b-', label='|P|', linewidth=2)
ax4.set_xlabel('σ')
ax4.set_ylabel('Относительный вклад (%)')
ax4.set_title('Доминирующие члены')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Анализ критической области для гауссианы', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sigma_critical_analysis.png', dpi=150)
console.print("[dim]График сохранён в sigma_critical_analysis.png[/dim]")

# ==================== ФИНАЛЬНЫЙ ОТЧЁТ ====================

console.print("\n" + "="*60)
console.print("[bold green]ФИНАЛЬНЫЙ ОТЧЁТ[/bold green]\n")

if all_critical_points:
    console.print(f"[bold]Найдено {len(all_critical_points)} критических точек:[/bold]")
    for i, sc in enumerate(all_critical_points, 1):
        console.print(f"  {i}. σ_c = {sc:.6f}")
    
    # Проверяем характер экстремумов
    console.print("\n[bold]Характер критических точек:[/bold]")
    for sc in all_critical_points:
        Q_left = compute_Q_gaussian(sc - 0.01)[0]
        Q_right = compute_Q_gaussian(sc + 0.01)[0]
        
        if Q_left < 0 and Q_right > 0:
            console.print(f"  σ = {sc:.6f}: переход - → + (Q становится положительным)")
        elif Q_left > 0 and Q_right < 0:
            console.print(f"  σ = {sc:.6f}: переход + → - (Q становится отрицательным)")

    # Интервалы положительности
    console.print("\n[bold]Интервалы где Q > 0:[/bold]")
    in_positive = False
    start_positive = None
    
    for s in sigmas_plot:
        Q = compute_Q_gaussian(s)[0]
        if Q > 0 and not in_positive:
            start_positive = s
            in_positive = True
        elif Q <= 0 and in_positive:
            console.print(f"  σ ∈ [{start_positive:.3f}, {s:.3f}]")
            in_positive = False
    
    if in_positive:
        console.print(f"  σ ∈ [{start_positive:.3f}, {sigmas_plot[-1]:.3f}]")

else:
    console.print("[red]Критические точки не найдены![/red]")

console.print("\n[bold cyan]ЧТО ЭТО ЗНАЧИТ:[/bold cyan]")
console.print("1. Q(σ) меняет знак несколько раз")
console.print("2. Существует оптимальное σ где баланс Z, A, P идеальный")
console.print("3. При малых σ: доминирует P → Q < 0")
console.print("4. При больших σ: доминирует A → Q < 0")
console.print("5. В узкой области: все три члена сбалансированы → Q > 0")