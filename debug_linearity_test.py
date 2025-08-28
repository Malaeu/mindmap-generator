#!/usr/bin/env python3
"""
DEBUG LINEARITY TEST
====================
The test is showing Q as linear when it shouldn't be
"""

import numpy as np
from fourier_conventions import compute_Q_weil
from rich.console import Console

console = Console()

ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832
]

def test_linearity_detailed():
    """Detailed test of Q linearity"""
    console.print("[bold cyan]DETAILED LINEARITY TEST[/bold cyan]\n")
    
    # Create two Gaussian pairs
    sigma1, sigma2 = 3.0, 5.0
    
    def h1(t, s=sigma1):
        return np.exp(-(t**2) / (2 * s**2))
    
    def hhat1(xi, s=sigma1):
        return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
    
    def h2(t, s=sigma2):
        return np.exp(-(t**2) / (2 * s**2))
    
    def hhat2(xi, s=sigma2):
        return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
    
    # Sum functions
    def h_sum(t):
        return h1(t) + h2(t)
    
    def hhat_sum(xi):
        return hhat1(xi) + hhat2(xi)
    
    # Compute Q values
    Q1, comp1 = compute_Q_weil(h1, hhat1, ZEROS, sigma_hint=sigma1, verbose=True)
    console.print()
    
    Q2, comp2 = compute_Q_weil(h2, hhat2, ZEROS, sigma_hint=sigma2, verbose=True)
    console.print()
    
    Q_sum, comp_sum = compute_Q_weil(h_sum, hhat_sum, ZEROS, sigma_hint=4.0, verbose=True)
    console.print()
    
    # Check linearity for each component
    console.print("[bold]LINEARITY CHECK BY COMPONENT:[/bold]")
    
    Z_linear = comp1['Z'] + comp2['Z']
    Z_actual = comp_sum['Z']
    console.print(f"Z: {Z_actual:.6f} vs {Z_linear:.6f} (diff: {abs(Z_actual - Z_linear):.6f})")
    
    A_linear = comp1['A'] + comp2['A']
    A_actual = comp_sum['A']
    console.print(f"A: {A_actual:.6f} vs {A_linear:.6f} (diff: {abs(A_actual - A_linear):.6f})")
    
    P_linear = comp1['P'] + comp2['P']
    P_actual = comp_sum['P']
    console.print(f"P: {P_actual:.6f} vs {P_linear:.6f} (diff: {abs(P_actual - P_linear):.6f})")
    
    Q_linear = Q1 + Q2
    console.print(f"\nQ: {Q_sum:.6f} vs {Q_linear:.6f} (diff: {abs(Q_sum - Q_linear):.6f})")
    
    # Test specific non-linearity: A and P should not be linear
    console.print("\n[bold]TESTING COMPONENT NON-LINEARITY:[/bold]")
    
    # For Z-term (should be linear)
    test_points = [14.134725, 21.022040, 25.010858]
    Z_test = 0
    for gamma in test_points:
        Z_test += h_sum(gamma)
    Z_test *= 2  # Account for ±γ
    
    Z_expected = 0
    for gamma in test_points:
        Z_expected += h1(gamma) + h2(gamma)
    Z_expected *= 2
    
    console.print(f"Z linearity test: {Z_test:.6f} = {Z_expected:.6f}? {abs(Z_test - Z_expected) < 1e-10}")
    
    # For A-term (should NOT be linear due to digamma)
    from scipy import special
    
    def arch_integrand(t, h_func):
        z = 0.25 + 0.5j * t
        psi = special.digamma(z)
        return h_func(t) * (np.real(psi) - np.log(np.pi)) / (2*np.pi)
    
    t_test = 5.0
    A_h1 = arch_integrand(t_test, h1)
    A_h2 = arch_integrand(t_test, h2)
    A_sum = arch_integrand(t_test, h_sum)
    
    console.print(f"A at t=5: {A_sum:.6f} vs {A_h1 + A_h2:.6f} (linear: {abs(A_sum - (A_h1 + A_h2)) < 1e-10})")
    
    # For P-term (should be linear in ĥ)
    test_log = np.log(7)  # Prime 7
    P_h1 = hhat1(test_log)
    P_h2 = hhat2(test_log)
    P_sum = hhat_sum(test_log)
    
    console.print(f"P at log(7): {P_sum:.6f} vs {P_h1 + P_h2:.6f} (linear: {abs(P_sum - (P_h1 + P_h2)) < 1e-10})")

def test_with_different_functions():
    """Test with very different functions to ensure non-linearity shows"""
    console.print("\n[bold cyan]TESTING WITH VERY DIFFERENT FUNCTIONS[/bold cyan]\n")
    
    # Use very different sigmas
    sigma1, sigma2 = 1.0, 10.0
    
    def h1(t, s=sigma1):
        return np.exp(-(t**2) / (2 * s**2))
    
    def hhat1(xi, s=sigma1):
        return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
    
    def h2(t, s=sigma2):
        return np.exp(-(t**2) / (2 * s**2))
    
    def hhat2(xi, s=sigma2):
        return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
    
    def h_sum(t):
        return h1(t) + h2(t)
    
    def hhat_sum(xi):
        return hhat1(xi) + hhat2(xi)
    
    Q1, comp1 = compute_Q_weil(h1, hhat1, ZEROS[:5], sigma_hint=sigma1, verbose=False)
    Q2, comp2 = compute_Q_weil(h2, hhat2, ZEROS[:5], sigma_hint=sigma2, verbose=False)
    Q_sum, comp_sum = compute_Q_weil(h_sum, hhat_sum, ZEROS[:5], sigma_hint=5.5, verbose=False)
    
    console.print(f"σ₁={sigma1}, σ₂={sigma2}")
    console.print(f"Q₁ = {Q1:.6f} (Z={comp1['Z']:.3f}, A={comp1['A']:.3f}, P={comp1['P']:.3f})")
    console.print(f"Q₂ = {Q2:.6f} (Z={comp2['Z']:.3f}, A={comp2['A']:.3f}, P={comp2['P']:.3f})")
    console.print(f"Q(h₁+h₂) = {Q_sum:.6f} (Z={comp_sum['Z']:.3f}, A={comp_sum['A']:.3f}, P={comp_sum['P']:.3f})")
    console.print(f"Q₁ + Q₂ = {Q1 + Q2:.6f}")
    console.print(f"\nDifference: |Q(h₁+h₂) - (Q₁+Q₂)| = {abs(Q_sum - (Q1 + Q2)):.6f}")
    
    is_linear = abs(Q_sum - (Q1 + Q2)) < 0.01
    
    if is_linear:
        console.print("[yellow]WARNING: Q still appears linear![/yellow]")
        console.print("This might be because Z, P are linear and A differences are small")
    else:
        console.print("[green]✅ Q is non-linear as expected[/green]")

def main():
    test_linearity_detailed()
    test_with_different_functions()
    
    console.print("\n[bold]CONCLUSION:[/bold]")
    console.print("Q(h) = Z(h) - A(h) - P(h)")
    console.print("- Z(h) is LINEAR in h (sum over zeros)")
    console.print("- P(h) is LINEAR in ĥ (sum over primes)")
    console.print("- A(h) is NON-LINEAR due to h(t) × ψ(1/4 + it/2) product")
    console.print("\nThe non-linearity is subtle and may be small for similar functions")

if __name__ == "__main__":
    main()