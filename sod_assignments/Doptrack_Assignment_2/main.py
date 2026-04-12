"""
main.py — run all tasks or a specific one.
    python main.py                 # runs everything, saves figures
    python main.py --task 1        # runs only task 1
    python main.py --no-plots      # skip all plots entirely
"""

import sys
import os
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _here,
    os.path.abspath(os.path.join(_here, '..')),
    os.path.abspath(os.path.join(_here, '../..')),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Extract archives
try:
    from utility_functions.data import extract_tar
    extract_tar(os.path.join(_here, '..', 'metadata.tar.xz'))
    extract_tar(os.path.join(_here, '..', 'data.tar.xz'))
except Exception as e:
    print(f"[warning] archive extraction: {e}")

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — never opens windows

import setup as setup_module
import estimation_utils as eu

from configs.task_1_arc_length import CONFIGS as T1
from configs.task_2_bias        import CONFIGS as T2
from configs.task_3_dynmodel    import CONFIGS as T3
from configs.task_4_passes      import CONFIGS as T4
from configs.task_5_validation  import CONFIGS as T5

ALL_TASKS = {
    1: ("Arc length sensitivity",      T1),
    2: ("Bias estimation on/off",      T2),
    3: ("Dynamical model sensitivity", T3),
    4: ("Pass selection",              T4),
    # 5: ("Validation vs TLE",           T5),
}

NB_ITERATIONS = 10
FIGURES_ROOT  = os.path.join(_here, 'figures')


def safe_folder_name(label):
    """Convert a config label to a safe folder name."""
    return label.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace(':', '')


def run_configs(configs, task_num, do_plots=True):
    results = []
    for cfg in configs:
        print(f"\n>>> Building estimator for: {cfg['label']}")
        built  = setup_module.build_estimator(cfg)
        iters  = cfg.get('nb_iterations', NB_ITERATIONS)
        result = eu.run_task(built, nb_iterations=iters, label=cfg['label'])

        if do_plots:
            fig_dir = os.path.join(FIGURES_ROOT,
                                   f'task{task_num}',
                                   safe_folder_name(cfg['label']))
            os.makedirs(fig_dir, exist_ok=True)
            eu.save_residuals(result, built, fig_dir)
            eu.save_residual_histogram(result, fig_dir)
            eu.save_rsw_keplerian(result, built, fig_dir)

        results.append(result)
    return results


def main(tasks_to_run=None, do_plots=True):
    all_results = []
    for task_num, (task_name, configs) in ALL_TASKS.items():
        if tasks_to_run and task_num not in tasks_to_run:
            continue
        print(f"\n{'#'*60}")
        print(f"  TASK {task_num}: {task_name}")
        print(f"{'#'*60}")
        results = run_configs(configs, task_num, do_plots=do_plots)
        all_results.extend(results)
        print(f"\n--- Task {task_num} summary ---")
        eu.summary_table(results)

    print(f"\n{'='*60}")
    print("  FULL SUMMARY")
    print(f"{'='*60}")
    eu.summary_table(all_results)

    if do_plots:
        summary_dir = os.path.join(FIGURES_ROOT, 'summary')
        os.makedirs(summary_dir, exist_ok=True)
        eu.save_summary_bar_chart(all_results, summary_dir,
                                  title='All configurations --- RMS and TLE distance')
        eu.save_rsw_bar_chart(all_results, summary_dir,
                              title='All configurations --- RSW RMS components')
        print(f"\n[figures saved to: {FIGURES_ROOT}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, nargs='+')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    main(tasks_to_run=args.task, do_plots=not args.no_plots)