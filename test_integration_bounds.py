#!/usr/bin/env python3
"""
TEST INTEGRATION BOUNDS CONSISTENCY
===================================
Check if integration bounds affect results
"""

import numpy as np
from scipy import integrate, special
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def test_archimedean_bounds():
    """Test if different integration bounds give consistent results"""
    console.print("[bold cyan]TESTING ARCHIMEDEAN INTEGRATION BOUNDS[/bold cyan]\n")
    
    def archimedean_integrand(t, sigma):
        h = np.exp(-(t**2) / (2 * sigma**2))
        z = 0.25 + 0.5j * t
        psi_val = special.digamma(z)
        return h * (np.real(psi_val) - np.log(np.pi)) / (2*np.pi)
    
    sigmas = [2.0, 5.0, 10.0, 15.0]
    bounds_multipliers = [5, 10, 15, 20, 30]
    
    table = Table(box=box.SIMPLE, title="Archimedean Term vs Integration Bounds")
    table.add_column("σ", justify="center")
    for mult in bounds_multipliers:
        table.add_column(f"t_max={mult}σ", justify="right")
    table.add_column("Δmax", justify="right")
    
    for sigma in sigmas:
        row = [f"{sigma:.1f}"]
        values = []
        
        for mult in bounds_multipliers:
            t_max = mult * sigma
            result, _ = integrate.quad(
                lambda t: archimedean_integrand(t, sigma),
                -t_max, t_max, 
                limit=400, epsrel=1e-10
            )
            values.append(result)
            row.append(f"{result:.6f}")
        
        # Calculate max difference
        delta_max = max(values) - min(values)
        row.append(f"{delta_max:.2e}")
        table.add_row(*row)
    
    console.print(table)
    
    # Check tail contribution
    console.print("\n[bold yellow]TAIL CONTRIBUTION ANALYSIS:[/bold yellow]")
    
    for sigma in [5.0, 10.0]:
        # Full integral with large bounds
        full, _ = integrate.quad(
            lambda t: archimedean_integrand(t, sigma),
            -30*sigma, 30*sigma,
            limit=400
        )
        
        # Core integral  
        core, _ = integrate.quad(
            lambda t: archimedean_integrand(t, sigma),
            -10*sigma, 10*sigma,
            limit=400
        )
        
        tail_contrib = abs(full - core)
        rel_tail = tail_contrib / abs(full) * 100
        
        console.print(f"σ={sigma}: tail beyond ±10σ contributes {tail_contrib:.2e} ({rel_tail:.2f}%)")

def test_prime_term_cutoff():
    """Test prime term convergence with different cutoffs"""
    console.print("\n[bold cyan]TESTING PRIME TERM CUTOFF[/bold cyan]\n")
    
    def sieve_primes(n_max):
        sieve = [True] * (n_max + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n_max**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n_max + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    sigmas = [3.0, 5.0, 10.0]
    cutoffs = [100, 500, 1000, 2000, 5000]
    
    table = Table(box=box.SIMPLE, title="Prime Term vs Cutoff")
    table.add_column("σ", justify="center")
    for cutoff in cutoffs:
        table.add_column(f"N_max={cutoff}", justify="right")
    table.add_column("Converged?", justify="center")
    
    for sigma in sigmas:
        def hhat(xi):
            return np.sqrt(2*np.pi) * sigma * np.exp(-(sigma**2) * (xi**2) / 2)
        
        row = [f"{sigma:.1f}"]
        values = []
        
        for N_max in cutoffs:
            primes = sieve_primes(N_max)
            P = 0.0
            
            for p in primes:
                log_p = np.log(p)
                P += (2 * (log_p / np.sqrt(p)) * hhat(log_p)) / (2*np.pi)
                
                # Powers of p
                pk = p * p
                while pk <= N_max:
                    P += (2 * (np.log(p) / np.sqrt(pk)) * hhat(np.log(pk))) / (2*np.pi)
                    pk *= p
            
            values.append(P)
            row.append(f"{P:.6f}")
        
        # Check convergence
        relative_changes = []
        for i in range(1, len(values)):
            if abs(values[i-1]) > 1e-10:
                rel_change = abs(values[i] - values[i-1]) / abs(values[i-1])
                relative_changes.append(rel_change)
        
        converged = all(rc < 0.001 for rc in relative_changes[-2:]) if len(relative_changes) >= 2 else False
        row.append("✅" if converged else "❌")
        
        table.add_row(*row)
    
    console.print(table)

def test_validation_bounds_mismatch():
    """Test if validation uses different bounds than computation"""
    console.print("\n[bold cyan]TESTING VALIDATION VS COMPUTATION BOUNDS[/bold cyan]\n")
    
    sigma = 5.0
    
    def h(t):
        return np.exp(-(t**2) / (2 * sigma**2))
    
    def hhat(xi):
        return np.sqrt(2*np.pi) * sigma * np.exp(-(sigma**2) * (xi**2) / 2)
    
    # Parseval with fixed bounds (like in validate_fourier_pair)
    integral_t_fixed, _ = integrate.quad(lambda t: np.abs(h(t))**2, -50, 50, limit=400)
    integral_xi_fixed, _ = integrate.quad(lambda xi: np.abs(hhat(xi))**2 / (2*np.pi), -50, 50, limit=400)
    
    # Parseval with adaptive bounds
    t_max = 10 * sigma
    integral_t_adaptive, _ = integrate.quad(lambda t: np.abs(h(t))**2, -t_max, t_max, limit=400)
    integral_xi_adaptive, _ = integrate.quad(lambda xi: np.abs(hhat(xi))**2 / (2*np.pi), -t_max, t_max, limit=400)
    
    console.print(f"Fixed bounds [-50, 50]:")
    console.print(f"  ∫|h|² dt = {integral_t_fixed:.6f}")
    console.print(f"  (1/2π)∫|ĥ|² dξ = {integral_xi_fixed:.6f}")
    console.print(f"  Error: {abs(integral_t_fixed - integral_xi_fixed):.2e}")
    
    console.print(f"\nAdaptive bounds [-{t_max:.0f}, {t_max:.0f}]:")
    console.print(f"  ∫|h|² dt = {integral_t_adaptive:.6f}")
    console.print(f"  (1/2π)∫|ĥ|² dξ = {integral_xi_adaptive:.6f}")
    console.print(f"  Error: {abs(integral_t_adaptive - integral_xi_adaptive):.2e}")
    
    console.print(f"\nDifference between methods:")
    console.print(f"  Δ(∫|h|²) = {abs(integral_t_fixed - integral_t_adaptive):.2e}")
    console.print(f"  Δ(∫|ĥ|²) = {abs(integral_xi_fixed - integral_xi_adaptive):.2e}")

def main():
    console.print("[bold cyan]INTEGRATION BOUNDS ANALYSIS[/bold cyan]")
    console.print("[dim]Checking numerical stability with different bounds[/dim]\n")
    
    test_archimedean_bounds()
    test_prime_term_cutoff()
    test_validation_bounds_mismatch()
    
    console.print("\n[bold green]RECOMMENDATIONS:[/bold green]")
    console.print("1. Archimedean: t_max = 10σ is sufficient for most cases")
    console.print("2. Prime term: N_max = 1000 gives good convergence")
    console.print("3. Validation should use adaptive bounds matching computation")

if __name__ == "__main__":
    main()