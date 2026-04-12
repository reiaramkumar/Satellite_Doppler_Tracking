"""
JUPYTER NOTEBOOK CELLS — Interactive Plots for Assignment 3
=============================================================

Copy-paste each section between the ═══ lines into a separate Jupyter cell.
Run them in order. Cell 1 sets up storage, Cell 2 is a helper you call after
each estimation, Cell 3+ generate the interactive plots.

Requirements: pip install plotly
(Plotly renders natively in Jupyter — no extensions needed)
"""

# ═══════════════════════════════════════════
#  CELL 1: Setup — run this once at the top
# ═══════════════════════════════════════════

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Storage for all configurations
all_results = {}

param_labels = ['x', 'y', 'z', 'ẋ', 'ẏ', 'ż', 'Cᴅ', 'μ']
param_labels_10 = param_labels + ['C₂₀', 'C₂₂']

# Colour palette — green=converged, amber=partial, red=diverged
COLORS = {
    'noise_0p1': '#2563EB',
    'noise_1p0': '#1D9E75',
    'noise_10': '#5DCAA5',
    'three_day_one_day_arcs': '#7F77DD',
    'manual_1km_1ms': '#E24B4A',
    'nearby_stations': '#85B7EB',
    'far': '#0F6E56',
    'c20_c22_single_station': '#D85A30',
    'c20_c22_far_stations': '#BA7517',
}

print("Setup done. Run your estimations, then call store() after each one.")


# ═══════════════════════════════════════════
#  CELL 2: Helper function — call after each estimation run
# ═══════════════════════════════════════════

def store_from_results(name, results, status='converged'):
    """
    Populate all_results from a results dict returned by run_scenario().

    Parameters
    ----------
    name : str
        Label for this configuration, e.g. 'Q1 baseline'
    results : dict
        The dict returned by run_scenario()
    status : str
        'converged', 'partial', or 'diverged'
    """
    raw = results['raw']
    formal_errors = raw['formal_errors']
    true_errors = raw['true_errors']
    covariance = raw['covariance']
    residual_history = raw['residual_history']
    observation_times = raw['observation_times']

    rms_per_iter = []
    try:
        for i in range(residual_history.shape[1]):
            rms_per_iter.append(float(np.sqrt(np.mean(residual_history[:, i] ** 2))))
    except Exception:
        pass

    try:
        cond = float(np.linalg.cond(covariance))
    except Exception:
        cond = None

    all_results[name] = {
        'formal': np.array(formal_errors).copy(),
        'true': np.array(true_errors).copy(),
        'ratio': np.abs(np.array(true_errors) / np.array(formal_errors)),
        'rms': rms_per_iter,
        'cond': cond,
        'color': COLORS.get(name),
        'status': status,
        'n_obs': len(observation_times),
    }

    print(f"Stored '{name}' — {len(formal_errors)} params, "
          f"{len(observation_times)} obs, status={status}")
    if cond:
        print(f"  Condition number: {cond:.2e}")


def store(name, status='converged'):
    """
    Call this right after each estimation completes.
    It grabs the variables already in your notebook's namespace.

    Parameters
    ----------
    name : str
        Label for this configuration, e.g. 'Q1 baseline'
    status : str
        'converged', 'partial', or 'diverged'

    Example
    -------
    After running the Q1 baseline estimation:
        store('Q1 baseline', 'converged')

    After Q3 (diverged):
        store('Q3 diverged', 'diverged')
    """
    # These variables should already exist from the assignment code
    rms_per_iter = []
    try:
        for i in range(residual_history.shape[1]):
            rms_per_iter.append(float(np.sqrt(np.mean(residual_history[:, i] ** 2))))
    except:
        pass

    try:
        cond = float(np.linalg.cond(covariance))
    except:
        cond = None

    all_results[name] = {
        'formal': formal_errors.copy(),
        'true': true_errors.copy(),
        'ratio': np.abs(true_errors / formal_errors),
        'rms': rms_per_iter,
        'cond': cond,
        'color': COLORS.get(name, '#888888'),
        'status': status,
        'n_obs': len(observation_times),
    }

    print(f"Stored '{name}' — {len(formal_errors)} params, "
          f"{len(observation_times)} obs, status={status}")
    if cond:
        print(f"  Condition number: {cond:.2e}")


# ═══════════════════════════════════════════
#  CELL 3: Formal errors comparison
# ═══════════════════════════════════════════

def plot_formal():
    fig = go.Figure()

    for name, r in all_results.items():
        n = len(r['formal'])
        labels = param_labels_10[:n]
        symbol = 'x' if r['status'] == 'diverged' else 'circle'
        size = 12 if r['status'] == 'diverged' else 9

        fig.add_trace(go.Scatter(
            x=labels, y=r['formal'],
            mode='markers', name=name,
            marker=dict(color=r['color'], symbol=symbol, size=size,
                        line=dict(width=1.5, color=r['color'])),
            hovertemplate='<b>%{x}</b>: %{y:.3e}<extra>' + name + '</extra>'
        ))

    fig.update_layout(
        title='Formal errors — all configurations',
        yaxis=dict(type='log', title='Formal error (log scale)'),
        xaxis_title='Parameter',
        template='plotly_white',
        height=450,
        hovermode='closest',
    )
    fig.show()
    return fig





# ═══════════════════════════════════════════
#  CELL 4: True-to-formal ratio comparison
# ═══════════════════════════════════════════

def plot_ratio():
    if not all_results:
        print("No results stored yet. Run store_from_results() first.")
        return
    fig = go.Figure()

    for name, r in all_results.items():
        n = len(r['ratio'])
        labels = param_labels_10[:n]
        symbol = 'x' if r['status'] == 'diverged' else 'circle'
        size = 12 if r['status'] == 'diverged' else 9

        fig.add_trace(go.Scatter(
            x=labels, y=r['ratio'],
            mode='markers', name=name,
            marker=dict(color=r['color'], symbol=symbol, size=size,
                        line=dict(width=1.5, color=r['color'])),
            hovertemplate='<b>%{x}</b>: %{y:.2f}<extra>' + name + '</extra>'
        ))

    # Ideal line
    fig.add_hline(y=1.0, line_dash='dash', line_color='#534AB7',
                  annotation_text='ideal = 1.0',
                  annotation_position='top right')

    # Danger zone
    fig.add_hrect(y0=1.5, y1=max(max(r['ratio']) for r in all_results.values()) * 1.1,
                  fillcolor='red', opacity=0.04, line_width=0)

    fig.update_layout(
        title='True-to-formal error ratio — all configurations',
        yaxis_title='|True error| / Formal error',
        xaxis_title='Parameter',
        template='plotly_white',
        height=450,
        hovermode='closest',
    )
    fig.show()
    return fig



# ═══════════════════════════════════════════
#  CELL 5: Convergence rate comparison
# ═══════════════════════════════════════════

def plot_convergence(noise_level=1.0):
    fig = go.Figure()

    for name, r in all_results.items():
        if not r['rms']:
            continue
        iters = list(range(1, len(r['rms']) + 1))
        symbol = 'x' if r['status'] == 'diverged' else 'circle'

        fig.add_trace(go.Scatter(
            x=iters, y=r['rms'],
            mode='lines+markers', name=name,
            line=dict(color=r['color'], width=2),
            marker=dict(color=r['color'], symbol=symbol, size=6),
            hovertemplate='Iter %{x}: RMS = %{y:.4f} m/s<extra>' + name + '</extra>'
        ))

    # Noise floor
    fig.add_hline(y=noise_level, line_dash='dash', line_color='#1D9E75',
                  opacity=0.6, annotation_text=f'noise floor ({noise_level} m/s)')

    fig.update_layout(
        title='Convergence rate — all configurations',
        yaxis=dict(type='log', title='RMS residual [m/s]'),
        xaxis_title='Iteration',
        template='plotly_white',
        height=450,
        hovermode='x unified',
    )
    fig.show()
    return fig



# ═══════════════════════════════════════════
#  CELL 6: Condition number bar chart
# ═══════════════════════════════════════════

def plot_condition():
    names = []
    conds = []
    colors = []

    for name, r in all_results.items():
        if r['cond'] is not None:
            names.append(name)
            conds.append(r['cond'])
            colors.append(r['color'])

    if not conds:
        print("No condition numbers available yet.")
        return

    fig = go.Figure(go.Bar(
        y=names, x=conds,
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: κ = %{x:.2e}<extra></extra>'
    ))

    fig.update_layout(
        title='Normal matrix conditioning',
        xaxis=dict(type='log', title='Condition number (log scale)'),
        template='plotly_white',
        height=max(250, len(names) * 40 + 100),
    )
    fig.show()
    return fig




# ═══════════════════════════════════════════
#  CELL 7: Three-panel dashboard
# ═══════════════════════════════════════════

def plot_dashboard(noise_level=1.0):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Formal errors', 'True/formal ratio', 'Convergence'),
        horizontal_spacing=0.07,
    )

    for name, r in all_results.items():
        n = len(r['formal'])
        labels = param_labels_10[:n]
        symbol = 'x' if r['status'] == 'diverged' else 'circle'

        # Col 1: formal errors
        fig.add_trace(go.Scatter(
            x=labels, y=r['formal'], mode='markers', name=name,
            marker=dict(color=r['color'], symbol=symbol, size=7,
                        line=dict(width=1, color=r['color'])),
            legendgroup=name, showlegend=True,
        ), row=1, col=1)

        # Col 2: ratio
        n_r = len(r['ratio'])
        fig.add_trace(go.Scatter(
            x=param_labels_10[:n_r], y=r['ratio'], mode='markers', name=name,
            marker=dict(color=r['color'], symbol=symbol, size=7,
                        line=dict(width=1, color=r['color'])),
            legendgroup=name, showlegend=False,
        ), row=1, col=2)

        # Col 3: convergence
        if r['rms']:
            iters = list(range(1, len(r['rms']) + 1))
            fig.add_trace(go.Scatter(
                x=iters, y=r['rms'], mode='lines+markers', name=name,
                line=dict(color=r['color'], width=1.5),
                marker=dict(color=r['color'], symbol=symbol, size=5),
                legendgroup=name, showlegend=False,
            ), row=1, col=3)

    fig.add_hline(y=1.0, line_dash='dash', line_color='#534AB7', row=1, col=2)
    fig.add_hline(y=noise_level, line_dash='dash', line_color='#1D9E75',
                  opacity=0.5, row=1, col=3)

    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=3)

    fig.update_layout(
        title='Orbit estimation — cross-configuration comparison',
        template='plotly_white',
        height=420, width=1100,
        legend=dict(font=dict(size=9), orientation='h', y=-0.18),
        hovermode='closest',
    )
    fig.show()
    return fig





# ═══════════════════════════════════════════
#  CELL 8: Summary table (prints markdown)
# ═══════════════════════════════════════════

def print_summary():
    from IPython.display import display, Markdown

    rows = "| Config | σ_pos [m] | σ_vel [m/s] | σ_Cd | σ_μ | κ(P) | N_obs | Status |\n"
    rows += "|--------|-----------|-------------|------|-----|------|-------|--------|\n"

    for name, r in all_results.items():
        f = r['formal']
        pos = np.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2)
        vel = np.sqrt(f[3] ** 2 + f[4] ** 2 + f[5] ** 2)
        cd = f[6]
        mu = f[7]
        cond = f"{r['cond']:.1e}" if r['cond'] else "—"
        rows += f"| {name} | {pos:.3f} | {vel:.6f} | {cd:.5f} | {mu:.2e} | {cond} | {r['n_obs']} | {r['status']} |\n"

    display(Markdown(rows))

    for name, r in all_results.items():
        n = len(r['formal'])
        print( param_labels_10[:n])
        print(f'formal error: {r['formal']} ratio: {r['ratio']} rms: {r['rms']}')






# ═══════════════════════════════════════════
#  CELL 9: Export static PNGs for LaTeX report
# ═══════════════════════════════════════════

# pip install kaleido    (needed for static export)

def export_all_png(out_dir='figures', noise_level=1.0, scale=2, width=800, height=450):
    """Export static PNG versions of all comparison plots.

    Parameters
    ----------
    out_dir : str
        Directory to write PNGs into (created if it doesn't exist).
    noise_level : float
        Noise floor line for the convergence plot (m/s).
    scale : int
        Resolution multiplier for kaleido (2 = 2x pixel density).
    width, height : int
        Base dimensions in pixels.

    Requires: pip install kaleido
    """
    import os
    import plotly.io as pio

    os.makedirs(out_dir, exist_ok=True)

    plots = [
        (plot_formal,                        'formal_errors'),
        (plot_ratio,                         'ratio'),
        (lambda: plot_convergence(noise_level), 'convergence'),
        (plot_condition,                     'condition'),
        (lambda: plot_dashboard(noise_level), 'dashboard'),
    ]

    for func, name in plots:
        fig = func()
        if fig is None:
            print(f"  Skipped '{name}' (no data or no figure returned).")
            continue
        path = os.path.join(out_dir, f'{name}.png')
        pio.write_image(fig, path, scale=scale, width=width, height=height)
        print(f"  Saved: {path}")

    print(f"\nAll plots exported to '{out_dir}/'.")


