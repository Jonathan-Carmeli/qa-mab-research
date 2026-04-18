"""
ablation_simulation.py
Ablation studies to understand what drives QA-MAB advantage.

Studies:
1. I_scale ablation: How does interference level affect the advantage?
2. B_scale ablation: Does skew in base utilities matter?
3. Lambda ablation: How does constraint penalty affect performance?
4. Tau sensitivity: How does tau schedule affect convergence?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation_core import NetworkEnvironment
from nb3r import NB3R
from qa_mab import QAMAB


def study_I_scale(N=20, m=4, T=500, n_runs=20):
    """How does interference scale affect QA vs NB3R?"""
    print("\n=== STUDY 1: Interference Scale ===")
    
    results = {}
    for I_scale in ['low', 'moderate', 'high']:
        print(f"\nI_scale = {I_scale}:")
        nb3r_vals = []
        qa_vals = []
        
        for r in range(n_runs):
            seed = 42 + r
            env_config = {'I_scale': I_scale}
            
            env_nb3r = NetworkEnvironment(N, m, seed=seed, **env_config)
            nb3r = NB3R(env_nb3r, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed)
            nb3r_hist = nb3r.run(T)
            nb3r_vals.append(np.mean(nb3r_hist[-50:]))
            
            env_qa = NetworkEnvironment(N, m, seed=seed, **env_config)
            qa = QAMAB(env_qa, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed)
            qa_hist = qa.run(T)
            qa_vals.append(np.mean(qa_hist[-50:]))
            
            print(f"  run {r+1}/{n_runs}  NB3R={nb3r_vals[-1]:.3f}  QA={qa_vals[-1]:.3f}")
        
        results[I_scale] = {
            'nb3r': (np.mean(nb3r_vals), np.std(nb3r_vals)),
            'qa': (np.mean(qa_vals), np.std(qa_vals)),
        }
    
    return results


def study_B_scale(N=20, m=4, T=500, n_runs=20):
    """Does skew in base utilities affect the advantage?"""
    print("\n=== STUDY 2: Base Utility Distribution ===")
    
    results = {}
    for B_scale in ['uniform', 'skewed']:
        print(f"\nB_scale = {B_scale}:")
        nb3r_vals = []
        qa_vals = []
        
        for r in range(n_runs):
            seed = 42 + r
            env_config = {'B_scale': B_scale}
            
            env_nb3r = NetworkEnvironment(N, m, seed=seed, **env_config)
            nb3r = NB3R(env_nb3r, tau0=0.1, delta_tau=0.05, alpha=0.3, seed=seed)
            nb3r_hist = nb3r.run(T)
            nb3r_vals.append(np.mean(nb3r_hist[-50:]))
            
            env_qa = NetworkEnvironment(N, m, seed=seed, **env_config)
            qa = QAMAB(env_qa, tau0=0.1, delta_tau=0.05, lambda_=0.5, seed=seed)
            qa_hist = qa.run(T)
            qa_vals.append(np.mean(qa_hist[-50:]))
            
            print(f"  run {r+1}/{n_runs}  NB3R={nb3r_vals[-1]:.3f}  QA={qa_vals[-1]:.3f}")
        
        results[B_scale] = {
            'nb3r': (np.mean(nb3r_vals), np.std(nb3r_vals)),
            'qa': (np.mean(qa_vals), np.std(qa_vals)),
        }
    
    return results


def study_lambda(N=20, m=4, T=500, n_runs=20):
    """How does lambda (constraint penalty) affect QA-MAB?"""
    print("\n=== STUDY 3: Lambda (Constraint Penalty) ===")
    
    results = {}
    for lambda_ in [0.1, 0.25, 0.5, 1.0, 2.0]:
        print(f"\nlambda = {lambda_}:")
        qa_vals = []
        
        for r in range(n_runs):
            seed = 42 + r
            env = NetworkEnvironment(N, m, seed=seed)
            qa = QAMAB(env, tau0=0.1, delta_tau=0.05, lambda_=lambda_, seed=seed)
            qa_hist = qa.run(T)
            qa_vals.append(np.mean(qa_hist[-50:]))
            
            print(f"  run {r+1}/{n_runs}  QA={qa_vals[-1]:.3f}")
        
        results[lambda_] = (np.mean(qa_vals), np.std(qa_vals))
    
    return results


def study_tau_schedule(N=20, m=4, T=500, n_runs=20):
    """How does tau schedule affect convergence?"""
    print("\n=== STUDY 4: Tau Schedule ===")
    
    results = {}
    for tau_config in [
        ('slow', 0.01),
        ('normal', 0.05),
        ('fast', 0.1),
    ]:
        name, delta_tau = tau_config
        print(f"\ntau schedule = {name} (delta_tau={delta_tau}):")
        nb3r_vals = []
        qa_vals = []
        
        for r in range(n_runs):
            seed = 42 + r
            
            env_nb3r = NetworkEnvironment(N, m, seed=seed)
            nb3r = NB3R(env_nb3r, tau0=0.1, delta_tau=delta_tau, alpha=0.3, seed=seed)
            nb3r_hist = nb3r.run(T)
            nb3r_vals.append(np.mean(nb3r_hist[-50:]))
            
            env_qa = NetworkEnvironment(N, m, seed=seed)
            qa = QAMAB(env_qa, tau0=0.1, delta_tau=delta_tau, lambda_=0.5, seed=seed)
            qa_hist = qa.run(T)
            qa_vals.append(np.mean(qa_hist[-50:]))
            
            print(f"  run {r+1}/{n_runs}  NB3R={nb3r_vals[-1]:.3f}  QA={qa_vals[-1]:.3f}")
        
        results[name] = {
            'nb3r': (np.mean(nb3r_vals), np.std(nb3r_vals)),
            'qa': (np.mean(qa_vals), np.std(qa_vals)),
        }
    
    return results


def main():
    N = 20
    m = 4
    T = 500
    n_runs = 20
    
    print(f"Ablation studies: N={N}, m={m}, T={T}, n_runs={n_runs}")
    
    # Run studies
    results_I = study_I_scale(N, m, T, n_runs)
    results_B = study_B_scale(N, m, T, n_runs)
    results_lambda = study_lambda(N, m, T, n_runs)
    results_tau = study_tau_schedule(N, m, T, n_runs)
    
    # Print summaries
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n1. Interference Scale:")
    print(f"   {'Scale':>10} | {'NB3R':>10} | {'QA-MAB':>10} | {'Advantage':>10}")
    print("   " + "-"*50)
    for scale, vals in results_I.items():
        adv = vals['qa'][0] - vals['nb3r'][0]
        print(f"   {scale:>10} | {vals['nb3r'][0]:>10.3f} | {vals['qa'][0]:>10.3f} | {adv:>10.3f}")
    
    print("\n2. Base Utility Distribution:")
    print(f"   {'Distribution':>12} | {'NB3R':>10} | {'QA-MAB':>10} | {'Advantage':>10}")
    print("   " + "-"*50)
    for scale, vals in results_B.items():
        adv = vals['qa'][0] - vals['nb3r'][0]
        print(f"   {scale:>12} | {vals['nb3r'][0]:>10.3f} | {vals['qa'][0]:>10.3f} | {adv:>10.3f}")
    
    print("\n3. Lambda (Constraint Penalty):")
    print(f"   {'Lambda':>10} | {'QA-MAB':>10} | {'Std':>10}")
    print("   " + "-"*35)
    for lambda_, vals in results_lambda.items():
        print(f"   {lambda_:>10.2f} | {vals[0]:>10.3f} | {vals[1]:>10.3f}")
    
    print("\n4. Tau Schedule:")
    print(f"   {'Schedule':>10} | {'NB3R':>10} | {'QA-MAB':>10} | {'Advantage':>10}")
    print("   " + "-"*50)
    for schedule, vals in results_tau.items():
        adv = vals['qa'][0] - vals['nb3r'][0]
        print(f"   {schedule:>10} | {vals['nb3r'][0]:>10.3f} | {vals['qa'][0]:>10.3f} | {adv:>10.3f}")


if __name__ == '__main__':
    main()
