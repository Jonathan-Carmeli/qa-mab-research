#!/usr/bin/env python3
"""Simple Cat 9 launcher — patches script in-memory and runs."""
import subprocess, sys, os

script_path = '/Users/jon_claw/qa-mab-research/simulations/physical_validation/validate_cat9_regret_convergence.py'
src = open(script_path).read()

# Fix path setup
src = src.replace(
    'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
    'sys.path.insert(0, "/Users/jon_claw/qa-mab-research")\nsys.path.insert(0, "/Users/jon_claw/qa-mab-research/simulations")'
)
# Fix imports
src = src.replace('from simulations.physical_env import', 'from physical_validation.physical_env import')
src = src.replace('from simulations.qa_mab_physical import', 'from physical_validation.qa_mab_physical import')
src = src.replace('from simulations.agents_physical.oracle_agent import', 'from physical_validation.agents_physical.oracle_agent import')
src = src.replace('from sa_solver_physical import', 'from physical_validation.sa_solver_physical import')

# Add reset_epoch to OptimalAgentBF class (after __init__)
src = src.replace(
    '    def update(self, *args): pass\n\nclass OracleSA:',
    '''    def update(self, *args): pass
    def reset_epoch(self, p):
        self.world.refresh_epoch(self.rng)

class OracleSA:'''
)
# Add reset_epoch to OracleSA
src = src.replace(
    '    def update(self, *args): pass\n\ndef run_single(',
    '''    def update(self, *args): pass
    def reset_epoch(self, p):
        self.world.refresh_epoch(self.rng)

def run_single('''
)

patched = '/tmp/validate_cat9_patched.py'
open(patched, 'w').write(src)

env = os.environ.copy()
env['PYTHONPATH'] = '/Users/jon_claw/qa-mab-research/simulations/physical_validation:/Users/jon_claw/qa-mab-research'

sys.exit(subprocess.call([sys.executable, patched], cwd='/Users/jon_claw/qa-mab-research', env=env))