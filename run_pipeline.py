"""Reproduce the complete four-part capstone from the repository root."""
import subprocess,sys
for script in ('generate_data.py','part_a.py','part_b.py','part_c.py','part_d.py'):
 print(f'=== {script} ===',flush=True)
 subprocess.run([sys.executable,script],check=True)