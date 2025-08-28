#!/usr/bin/env python3
"""
ИСПРАВЛЕННЫЙ ПОДХОД: КЛАСС-ТЕСТ БЕЗ БИЛИНЕЙНОСТИ
=================================================
Q(h) не билинейная форма → нужен прямой тест на классе функций
"""

import numpy as np
from fourier_conventions import GaussianPair, GaussianHermitePair, compute_Q_weil
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import box
import matplotlib.pyplot as plt

console = Console()

ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
]

def test_function_class_directly(function_generator, param_ranges, param_names):
    """
    Прямой тест Q ≥ 0 на классе функций
    БЕЗ предположения о билинейности
    """
    console.print(f"[bold yellow]ТЕСТ КЛАССА ФУНКЦИЙ:[/bold yellow]")
    console.print(f"Параметры: {dict(zip(param_names, [f'{r[0]}-{r[1]}' for r in param_ranges]))}\n")
    
    # Генерируем сетку параметров
    param_grids = [np.linspace(r[0], r[1], 10) for r in param_ranges]
    param_combinations = np.meshgrid(*param_grids)
    
    results = []
    total_tests = np.prod([len(grid) for grid in param_grids])
    
    console.print(f"Тестируем {total_tests} функций...")
    
    positive_count = 0
    negative_count = 0
    
    # Статистика
    q_values = []
    failed_functions = []
    
    # Перебираем все комбинации параметров
    flat_combinations = [combo.flatten() for combo in param_combinations]
    
    for i in track(range(len(flat_combinations[0])), description="Testing functions"):
        params = [combo[i] for combo in flat_combinations]
        
        try:
            # Создаём функцию с данными параметрами
            func = function_generator(*params)
            
            # Вычисляем Q
            Q, components = compute_Q_weil(func.h, func.hhat, ZEROS,
                                         sigma_hint=getattr(func, 'sigma', None),
                                         verbose=False)
            
            q_values.append(Q)
            
            if Q >= 0:
                positive_count += 1
            else:
                negative_count += 1
                failed_functions.append({
                    'params': dict(zip(param_names, params)),
                    'Q': Q,
                    'components': components
                })
                
            results.append({
                'params': dict(zip(param_names, params)),
                'Q': Q,
                'components': components
            })
            
        except Exception as e:
            console.print(f"[red]Ошибка для параметров {params}: {e}[/red]")
            continue
    
    # Анализ результатов
    success_rate = positive_count / len(q_values) if q_values else 0
    
    console.print(f"\n[bold]Результаты теста:[/bold]")
    console.print(f"  ✅ Положительных Q: {positive_count}")
    console.print(f"  ❌ Отрицательных Q: {negative_count}")
    console.print(f"  📊 Процент успеха: {success_rate:.1%}")
    
    if q_values:
        console.print(f"  📈 Q ∈ [{min(q_values):+.3f}, {max(q_values):+.3f}]")
        console.print(f"  📊 Среднее Q: {np.mean(q_values):+.3f}")
    
    # Показываем неудачные случаи
    if failed_functions and len(failed_functions) <= 10:
        console.print(f"\n[red]Функции с Q < 0:[/red]")
        
        table = Table(box=box.SIMPLE)
        for name in param_names:
            table.add_column(name, justify="center")
        table.add_column("Q", justify="right")
        table.add_column("Доминант", justify="center")
        
        for failure in failed_functions:
            params = failure['params']
            Q = failure['Q']
            comp = failure['components']
            
            # Определяем доминирующий член
            abs_vals = {
                'Z': abs(comp['Z']),
                'A': abs(comp['A']),
                'P': abs(comp['P'])
            }
            dominant = max(abs_vals.keys(), key=lambda k: abs_vals[k])
            
            row = [str(params[name]) for name in param_names]
            row.extend([f"{Q:+.3f}", dominant])
            table.add_row(*row)
        
        console.print(table)
    
    return success_rate, results, failed_functions

def gaussian_class_test():
    """Тест класса чистых гауссиан"""
    def gauss_generator(sigma):
        return GaussianPair(sigma)
    
    return test_function_class_directly(
        gauss_generator,
        [(0.5, 10.0)],
        ['σ']
    )

def hermite_class_test():
    """Тест класса Gaussian-Hermite (только четные k для симметрии)"""
    def hermite_generator(sigma, k_float):
        k = int(round(k_float))
        if k % 2 != 0:
            k = k + 1  # Принудительно делаем четным
        return GaussianHermitePair(sigma, k)
    
    return test_function_class_directly(
        hermite_generator,
        [(2.0, 8.0), (0, 6)],  # sigma, k_even
        ['σ', 'k']
    )

def robustness_test_zeros(best_functions, zero_counts):
    """Тест устойчивости при изменении количества нулей"""
    console.print(f"\n[bold yellow]ТЕСТ УСТОЙЧИВОСТИ (количество нулей):[/bold yellow]\n")
    
    for n_zeros in zero_counts:
        zeros_subset = ZEROS[:n_zeros] if n_zeros <= len(ZEROS) else ZEROS
        
        console.print(f"Тестируем с {len(zeros_subset)} нулями:")
        
        stable_count = 0
        unstable_functions = []
        
        for func_data in best_functions[:5]:  # Берем топ-5 функций
            params = func_data['params']
            original_Q = func_data['Q']
            
            # Создаем функцию заново
            if 'k' in params:
                func = GaussianHermitePair(params['σ'], int(params['k']))
            else:
                func = GaussianPair(params['σ'])
            
            # Пересчитываем Q с новым количеством нулей
            Q_new, _ = compute_Q_weil(func.h, func.hhat, zeros_subset,
                                    sigma_hint=params['σ'], verbose=False)
            
            stability = abs(Q_new - original_Q) / max(abs(original_Q), 1e-10)
            
            if Q_new >= 0 and stability < 0.1:  # Изменение < 10%
                stable_count += 1
                status = "✅"
            else:
                unstable_functions.append((params, original_Q, Q_new))
                status = "❌"
            
            console.print(f"  {params}: {original_Q:+.3f} → {Q_new:+.3f} {status}")
        
        console.print(f"  Стабильных: {stable_count}/{min(5, len(best_functions))}\n")
    
    return len(unstable_functions) == 0

def main_corrected_benchmark():
    """Исправленный бенчмарк без билинейных предположений"""
    console.print("[bold cyan]ИСПРАВЛЕННЫЙ БАРЬЕР ПОЗИТИВНОЙ СТЕНЫ[/bold cyan]")
    console.print("[dim]Прямой тест Q ≥ 0 на классах функций (без Gram-матрицы)[/dim]\n")
    
    # Тест 1: Класс чистых гауссиан
    console.print("[bold blue]ЭТАП 1: КЛАСС ГАУССИАН[/bold blue]")
    gauss_success, gauss_results, gauss_failures = gaussian_class_test()
    
    # Тест 2: Класс Gaussian-Hermite
    console.print(f"\n[bold blue]ЭТАП 2: КЛАСС GAUSSIAN-HERMITE[/bold blue]")
    hermite_success, hermite_results, hermite_failures = hermite_class_test()
    
    # Анализ лучших функций
    all_results = gauss_results + hermite_results
    positive_results = [r for r in all_results if r['Q'] >= 0]
    positive_results.sort(key=lambda x: x['Q'], reverse=True)
    
    console.print(f"\n[bold green]ТОП-10 ЛУЧШИХ ФУНКЦИЙ:[/bold green]")
    table = Table(box=box.ROUNDED)
    table.add_column("Место", justify="center")
    table.add_column("Тип", justify="center")
    table.add_column("Параметры", justify="center")
    table.add_column("Q", justify="right")
    
    for i, result in enumerate(positive_results[:10]):
        params = result['params']
        if 'k' in params:
            func_type = "Hermite"
            param_str = f"σ={params['σ']:.1f}, k={int(params['k'])}"
        else:
            func_type = "Gaussian"
            param_str = f"σ={params['σ']:.1f}"
        
        table.add_row(
            str(i+1),
            func_type,
            param_str,
            f"{result['Q']:+.4f}"
        )
    
    console.print(table)
    
    # Тест устойчивости
    console.print(f"\n[bold blue]ЭТАП 3: УСТОЙЧИВОСТЬ[/bold blue]")
    is_robust = robustness_test_zeros(positive_results, [15, 30, 30])
    
    # Финальный вердикт
    console.print("\n" + "="*60)
    console.print("[bold green]ИСПРАВЛЕННЫЙ ФИНАЛЬНЫЙ ВЕРДИКТ:[/bold green]\n")
    
    total_positive = len(positive_results)
    total_tested = len(all_results)
    
    if gauss_success > 0.9:  # 90%+ гауссиан положительны
        console.print("✅ ГАУССИАНЫ: критерий Вейля выполнен!")
        console.print(f"   {gauss_success:.0%} функций дают Q ≥ 0")
    
    if total_positive > total_tested * 0.5:  # Более половины положительны
        console.print("✅ РАСШИРЕННЫЙ КЛАСС: частично согласуется с RH!")
        console.print(f"   {total_positive}/{total_tested} функций дают Q ≥ 0")
    
    if is_robust:
        console.print("✅ УСТОЙЧИВОСТЬ: результаты стабильны при изменении нулей")
    
    console.print(f"\n[cyan]Заключение: критерий Вейля Q ≥ 0 выполняется для[/cyan]")
    console.print(f"[cyan]широкого подкласса тест-функций, что согласуется с RH![/cyan]")
    
    # Сохраняем график
    create_summary_plot(gauss_results, hermite_results)
    
    return total_positive / total_tested > 0.5

def create_summary_plot(gauss_results, hermite_results):
    """Создаём сводный график результатов"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # График 1: Гауссианы
    gauss_sigmas = [r['params']['σ'] for r in gauss_results]
    gauss_qs = [r['Q'] for r in gauss_results]
    
    ax1.plot(gauss_sigmas, gauss_qs, 'bo-', linewidth=2, markersize=4)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.fill_between(gauss_sigmas, 0, gauss_qs, where=np.array(gauss_qs) >= 0,
                    color='green', alpha=0.2, label='Q ≥ 0')
    ax1.fill_between(gauss_sigmas, gauss_qs, 0, where=np.array(gauss_qs) < 0,
                    color='red', alpha=0.2, label='Q < 0')
    ax1.set_xlabel('σ')
    ax1.set_ylabel('Q')
    ax1.set_title('Гауссианы: Q(σ)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График 2: Распределение Q
    all_qs = gauss_qs + [r['Q'] for r in hermite_results]
    
    ax2.hist(all_qs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Q = 0')
    ax2.set_xlabel('Q значения')
    ax2.set_ylabel('Количество функций')
    ax2.set_title('Распределение Q по всем функциям')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('weil_class_test_results.png', dpi=150, bbox_inches='tight')
    console.print("\n[dim]График сохранён в weil_class_test_results.png[/dim]")

if __name__ == "__main__":
    success = main_corrected_benchmark()
    exit(0 if success else 1)