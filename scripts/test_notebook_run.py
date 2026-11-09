import json
import matplotlib
matplotlib.use('Agg')

with open('M2_ReID_Demonstration.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Code cells: {len(code_cells)}")

scope = {}
all_passed = True
for i, c in enumerate(code_cells):
    code = ''.join(c['source'])
    code_lines = [line for line in code.split('\n') if not line.strip().startswith('!')]
    clean_code = '\n'.join(code_lines)
    try:
        exec(clean_code, scope)
        print(f"Cell {i+1}: OK")
    except Exception as e:
        print(f"Cell {i+1} ERROR: {e}")
        all_passed = False

if all_passed:
    print("ALL CELLS EXECUTED SUCCESSFULLY!")

