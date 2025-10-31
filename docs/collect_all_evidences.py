import glob, re, csv, numpy as np

files = sorted(glob.glob('outdir_real_noise_pe_study/seed_*/seed_*_evidences.txt'))

rows = []
for fp in files:
    try:
        vals = np.loadtxt(fp, comments='#').ravel()
        seed = int(re.search(r'/seed_(\d+)/', fp).group(1))
        rows.append([seed] + vals.tolist())
    except Exception as e:
        print(f"[skip] {fp}: {e}")

rows.sort(key=lambda x: x[0])

with open('aggregated_evidences_2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed', 'logz_welch', 'logz_sgvb', 'logBF_sgvb_vs_welch', 'BF_sgvb_vs_welch'])
    writer.writerows(rows)
    
    