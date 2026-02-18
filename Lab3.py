# =============================================================================
# ENABLING INTERACTIVE PLOTTING
# =============================================================================
# This magic command is REQUIRED for interactivity (Zoom/Pan)
# If this fails, try: %matplotlib notebook
%matplotlib widget 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. METHOD DEFINITIONS
# =============================================================================

def newton_raphson(func, deriv, x0, tol=1e-6, max_iter=50):
    history = []
    x_curr = x0
    for i in range(max_iter):
        f_val = func(x_curr)
        f_prime = deriv(x_curr)
        if abs(f_prime) < 1e-12: break 
        x_next = x_curr - (f_val / f_prime)
        error = abs(x_next - x_curr)
        history.append({"Iter": i+1, "Method": "Newton", "x_curr": x_curr, "f(x)": f_val, "Error": error})
        x_curr = x_next
        if error < tol: break
    return x_curr, pd.DataFrame(history)

def secant_method(func, x0, x1, tol=1e-6, max_iter=50):
    history = []
    x_prev, x_curr = x0, x1
    for i in range(max_iter):
        f_curr, f_prev = func(x_curr), func(x_prev)
        if abs(f_curr - f_prev) < 1e-12: break
        x_next = x_curr - f_curr * ((x_curr - x_prev) / (f_curr - f_prev))
        error = abs(x_next - x_curr)
        history.append({"Iter": i+1, "Method": "Secant", "x_curr": x_curr, "f(x)": f_curr, "Error": error})
        x_prev, x_curr = x_curr, x_next
        if error < tol: break
    return x_curr, pd.DataFrame(history)

def false_position(func, a, b, tol=1e-6, max_iter=50):
    history = []
    if func(a) * func(b) >= 0: return None, None 

    c_old = a
    for i in range(max_iter):
        fa, fb = func(a), func(b)
        c = b - fb * ((b - a) / (fb - fa))
        fc = func(c)
        error = abs(c - c_old)
        history.append({"Iter": i + 1, "Method": "FalsePos", "a": a, "b": b, "root": c, "f(x)": fc, "Error": error})
        if error < tol: break
        if fa * fc < 0: b = c
        else: a = c
        c_old = c
    return c, pd.DataFrame(history)

# =============================================================================
# 2. PLOTTING FUNCTION (Enhanced for Interaction)
# =============================================================================
def plot_interactive_false_position(func, a, b, root, label):
    # Create a new figure for each plot
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # 1. Generate Data
    margin = 1.0
    x_plot = np.linspace(a - margin, b + margin, 200)
    y_plot = func(x_plot)

    # 2. Plot Elements
    ax.plot(x_plot, y_plot, label='f(x)', color='blue', linewidth=2)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)
    
    # Chord & Bracket
    fa, fb = func(a), func(b)
    ax.plot([a, b], [fa, fb], '-.', color='purple', alpha=0.7, label='Chord')
    ax.hlines(0, a, b, colors='thistle', linestyles='solid', linewidth=6, label='Bracket')
    
    # Markers
    ax.plot(a, fa, 'kv', markersize=8, label='Lower Bound')
    ax.plot(b, fb, 'k^', markersize=8, label='Upper Bound')
    ax.plot(root, 0, 'ro', markersize=10, label=f'Root ({root:.4f})')
    
    ax.set_title(f"False Position: {label}")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()
    
    # Explicitly show the plot to trigger the widget
    plt.show()

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    functions_data = {
        "A": [lambda x: x**5 + x - 1, lambda x: 5*x**4 + 1, 0.0, 1.0, "x^5 + x - 1"],
        "B": [lambda x: x**5 + 5*x**3 - 4*x + 1, lambda x: 5*x**4 + 15*x**2 - 4, -1.0, 1.0, "x^5 + 5x^3 - 4x + 1"],
        "C": [lambda x: x**5 + 2*x**4 - x - 3, lambda x: 5*x**4 + 8*x**3 - 1, 1.0, 2.0, "x^5 + 2x^4 - x - 3"],
        "D": [lambda x: x**5 - 10, lambda x: 5*x**4, 1.0, 2.0, "x^5 - 10"],
        "E": [lambda x: x**5 + 2*x**2 + x - 0.5, lambda x: 5*x**4 + 4*x + 1, 0.0, 1.0, "x^5 + 2x^2 + x - 0.5"]
    }

    print("="*60)
    print(" NOTE: TO ZOOM WITH SCROLL WHEEL:")
    print(" 1. Run this cell.")
    print(" 2. In the graph toolbar, CLICK the 'Zoom' icon (Magnifying Glass).")
    print(" 3. Now SCROLL inside the graph area.")
    print("="*60 + "\n")

    for key, (func, deriv, a, b, label) in functions_data.items():
        print(f"--- Processing Function {key}: {label} ---")
        
        # Run False Position
        root_f, tab_f = false_position(func, a, b)
        
        if tab_f is not None:
            print(f"Root found at: {root_f:.6f}")
            print(tab_f.to_string(index=False))
            
            # Plot
            plot_interactive_false_position(func, a, b, root_f, label)
        else:
            print("False Position Failed (Invalid Bracket)")
            
        print("\n" + "-"*40 + "\n")
