#!/usr/bin/env python3
"""
ОТЛАДКА ЗНАКОВ В ФОРМУЛЕ ВЕЙЛЯ
================================
Проверяем правильность знаков в каждом члене
"""

import numpy as np
from scipy import special
from rich.console import Console

console = Console()

# Тестовый нуль
gamma_test = 14.134725

# Простая гауссиана
def gaussian(t, sigma=1.0):
    return np.exp(-t**2 / (2 * sigma**2))

def gaussian_hat(xi, sigma=1.0):
    return np.sqrt(2*np.pi) * sigma * np.exp(-sigma**2 * xi**2 / 2)

# Проверяем каждый член отдельно
console.print("[bold cyan]ПРОВЕРКА ЗНАКОВ В ФОРМУЛЕ ВЕЙЛЯ[/bold cyan]\n")

# 1. Член с нулями
# По теории: должен быть положительным для гауссианы на нулях
phi_at_gamma = gaussian(gamma_test)
console.print(f"1. φ(γ) = φ({gamma_test:.2f}) = {phi_at_gamma:.6f}")
console.print(f"   [green]Знак: + (правильно, φ > 0 для гауссианы)[/green]\n")

# 2. Архимедов член
# A(φ) = (1/2π) ∫ φ(t) Re[ψ(1/4 + it/2)] dt - log(π) φ(0)
t_test = 0.0
z = 0.25 + 0.5j * t_test
psi_val = special.digamma(z)
weight_at_zero = np.real(psi_val) - np.log(np.pi)
console.print(f"2. Вес в t=0: Re[ψ(1/4)] - log(π) = {weight_at_zero:.6f}")

# Проверим знак интеграла
from scipy import integrate
def arch_integrand(t):
    z = 0.25 + 0.5j * t
    psi = special.digamma(z)
    return gaussian(t) * np.real(psi) / (2 * np.pi)

integral, _ = integrate.quad(arch_integrand, -20, 20)
const_term = -np.log(np.pi) * gaussian(0)
A_total = integral + const_term

console.print(f"   Интеграл: {integral:.6f}")
console.print(f"   -log(π)φ(0): {const_term:.6f}")
console.print(f"   A(φ) = {A_total:.6f}")
console.print(f"   [yellow]Знак: {'+' if A_total > 0 else '-'}[/yellow]\n")

# 3. Член с простыми
# P(φ) = 2 Σ [Λ(n)/√n] φ̂(log n)
console.print("3. Первые члены суммы по простым:")
for p in [2, 3, 5, 7, 11]:
    log_p = np.log(p)
    phi_hat_val = gaussian_hat(log_p)
    contribution = 2 * (np.log(p) / np.sqrt(p)) * phi_hat_val
    console.print(f"   p={p}: 2*Λ({p})/√{p} * φ̂(log {p}) = {contribution:.6f}")

# Полная сумма
P_sum = 0.0
for n in range(2, 1000):
    # Упрощённо: только простые числа
    is_prime = all(n % d != 0 for d in range(2, int(np.sqrt(n)) + 1))
    if is_prime:
        lambda_n = np.log(n)
        log_n = np.log(n)
        phi_hat_val = gaussian_hat(log_n)
        P_sum += 2 * (lambda_n / np.sqrt(n)) * phi_hat_val

console.print(f"   P(φ) ≈ {P_sum:.6f}")
console.print(f"   [yellow]Знак: + (всегда положительный для гауссианы)[/yellow]\n")

# 4. Формула Вейля
console.print("[bold]ПРОВЕРКА ФОРМУЛЫ:[/bold]")
console.print("Стандартная форма: Σ_ρ φ(γ) = A(φ) + P(φ)")
console.print("Критерий Вейля: Q(φ) = Σ_ρ φ(γ) - A(φ) - P(φ)")
console.print("\nЕсли RH верна, то Q(φ) ≡ 0 для допустимых φ")
console.print("Но численно мы получаем Q < 0...\n")

# Альтернативная интерпретация
console.print("[bold red]ВОЗМОЖНАЯ ПРОБЛЕМА:[/bold red]")
console.print("1. Неполная сумма по нулям (только первые 150)")
console.print("2. Отсутствие тривиальных нулей ζ(-2n)")
console.print("3. Возможная ошибка в нормировке преобразования Фурье")

# Проверяем альтернативную нормировку
console.print("\n[bold]АЛЬТЕРНАТИВНЫЕ НОРМИРОВКИ:[/bold]")

# Вариант 1: унитарная нормировка
phi_hat_unitary = lambda xi: (1/np.sqrt(2*np.pi)) * integrate.quad(
    lambda t: gaussian(t) * np.exp(-1j*xi*t), -20, 20
)[0]

# Вариант 2: без 2π в знаменателе P
P_alt = P_sum * 2 * np.pi
console.print(f"P без 1/(2π): {P_alt:.6f}")

# Тест баланса
Z_approx = 0  # Для гауссианы с большой σ вклад нулей мал
Q_standard = Z_approx - A_total - P_sum
Q_alt = Z_approx - A_total - P_alt

console.print(f"\nQ (стандарт): {Q_standard:.6f}")
console.print(f"Q (альт. P): {Q_alt:.6f}")

# Проверка симметрии
console.print("\n[bold]ПРОВЕРКА СИММЕТРИИ:[/bold]")
console.print("Для вещественной чётной φ:")
console.print("- φ̂ также вещественная и чётная")
console.print("- Вклад ρ и ρ̄ одинаковый")
console.print("- P(φ) всегда положительный")

# Финальный вывод
console.print("\n[bold green]ВЫВОД:[/bold green]")
console.print("Знаки членов корректные, но баланс нарушен.")
console.print("Вероятные причины:")
console.print("1. Недостаточно нулей в сумме")
console.print("2. Нужна поправка на хвост распределения нулей")
console.print("3. Возможно, нужны тривиальные нули")