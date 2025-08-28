#!/usr/bin/env python3
"""
VERIFY CLOSURE FIX IS WORKING PROPERLY
======================================
Tests that the fixed functions with default parameters work correctly
when called from compute_Q_weil
"""

import numpy as np
from fourier_conventions import compute_Q_weil
from rich.console import Console

console = Console()

ZEROS = [14.134725, 21.022040, 25.010858]

def test_closure_fix_integration():
    """Test that functions with default params work in compute_Q_weil"""
    console.print("[bold cyan]TESTING CLOSURE FIX IN FULL INTEGRATION[/bold cyan]\n")
    
    test_sigmas = [2.0, 5.0, 8.0]
    
    # Store functions to test they maintain their sigma
    stored_functions = []
    
    for sigma in test_sigmas:
        # Create functions with closure fix (default parameter)
        def h(t, s=sigma):
            return np.exp(-(t**2) / (2 * s**2))
        
        def hhat(xi, s=sigma):
            return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
        
        # Store for later verification
        stored_functions.append((sigma, h, hhat))
        
        # Test immediate use
        Q, components = compute_Q_weil(h, hhat, ZEROS, sigma_hint=sigma, verbose=False)
        console.print(f"σ={sigma}: Q = {Q:+.6f}, Z = {components['Z']:+.6f}")
    
    console.print("\n[bold yellow]VERIFYING STORED FUNCTIONS RETAIN SIGMA:[/bold yellow]")
    
    all_passed = True
    # Critical test: functions should retain their sigma even after loop
    for sigma, h_stored, hhat_stored in stored_functions:
        # Test at t=1.0
        h_value = h_stored(1.0)
        expected = np.exp(-1.0 / (2 * sigma**2))
        error = abs(h_value - expected)
        
        if error >= 1e-10:
            all_passed = False
        
        status = "✅" if error < 1e-10 else "❌"
        console.print(f"σ={sigma}: h(1) = {h_value:.6f}, expected = {expected:.6f}, error = {error:.2e} {status}")
        
        # Also compute Q to ensure full integration works
        Q, components = compute_Q_weil(h_stored, hhat_stored, ZEROS, sigma_hint=sigma, verbose=False)
        
        if abs(components['Z']) < 0.001:  # Z should be non-zero for these sigmas
            console.print(f"  [red]WARNING: Z ≈ 0 for σ={sigma}, might indicate function not working![/red]")
            if sigma > 3.0:  # For larger sigmas, Z should definitely be non-zero
                all_passed = False
    
    return all_passed

def test_calling_conventions():
    """Test that h_func and hhat_func are called correctly"""
    console.print("\n[bold yellow]TESTING FUNCTION CALLING CONVENTIONS:[/bold yellow]")
    
    call_count = {'h': 0, 'hhat': 0}
    
    sigma_test = 5.0
    
    def h_instrumented(t, s=sigma_test):
        call_count['h'] += 1
        return np.exp(-(t**2) / (2 * s**2))
    
    def hhat_instrumented(xi, s=sigma_test):  
        call_count['hhat'] += 1
        return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
    
    # Run compute_Q_weil
    Q, components = compute_Q_weil(h_instrumented, hhat_instrumented, ZEROS[:5], 
                                   sigma_hint=sigma_test, verbose=False)
    
    console.print(f"h called: {call_count['h']} times")
    console.print(f"hhat called: {call_count['hhat']} times")
    console.print(f"Q = {Q:+.6f}")
    
    if call_count['h'] == 0 or call_count['hhat'] == 0:
        console.print("[red]ERROR: Functions not being called![/red]")
        return False
    
    console.print("[green]✅ Functions called successfully[/green]")
    return True

def test_parameter_shadowing():
    """Test for parameter shadowing issues"""
    console.print("\n[bold yellow]TESTING FOR PARAMETER SHADOWING:[/bold yellow]")
    
    # This tests if inner functions accidentally shadow outer parameters
    sigma_outer = 10.0
    
    def create_funcs(sigma):
        # Potential bug: if sigma is not captured properly
        def h(t, s=sigma):  # s captures sigma
            return np.exp(-(t**2) / (2 * s**2))
        
        def hhat(xi, s=sigma):
            return np.sqrt(2*np.pi) * s * np.exp(-(s**2) * (xi**2) / 2)
        
        return h, hhat
    
    # Create functions with different sigmas
    h1, hhat1 = create_funcs(3.0)
    h2, hhat2 = create_funcs(7.0)
    
    # Test they give different values
    val1 = h1(1.0)
    val2 = h2(1.0)
    
    console.print(f"h1(1) with σ=3: {val1:.6f}")
    console.print(f"h2(1) with σ=7: {val2:.6f}")
    
    if abs(val1 - val2) < 0.01:
        console.print("[red]ERROR: Functions giving same values - closure not working![/red]")
        return False
    
    console.print("[green]✅ No parameter shadowing detected[/green]")
    return True

def main():
    console.print("[bold cyan]CLOSURE FIX VERIFICATION SUITE[/bold cyan]")
    console.print("[dim]Ensuring the default parameter fix works correctly[/dim]\n")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1
    if test_closure_fix_integration():
        tests_passed += 1
    
    # Test 2  
    if test_calling_conventions():
        tests_passed += 1
    
    # Test 3
    if test_parameter_shadowing():
        tests_passed += 1
    
    # Summary
    console.print("\n" + "="*50)
    console.print(f"[bold]RESULTS: {tests_passed}/{total_tests} tests passed[/bold]")
    
    if tests_passed == total_tests:
        console.print("[bold green]✅ CLOSURE FIX VERIFIED WORKING![/bold green]")
        return True
    else:
        console.print("[bold red]❌ CLOSURE FIX HAS ISSUES![/bold red]")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)