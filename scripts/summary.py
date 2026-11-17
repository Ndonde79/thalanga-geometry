import json, glob, os
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'out', '*_info.json')))]
keys = ['domain', 'n_points', 'MS_fit_s', 'OH_fit_s', 'MS_honour_accuracy', 'OH_honour_accuracy', 'MS_volume_Mm3', 'OH_volume_Mm3',
        'MS_true_thickness_m_p50_p90', 'OH_true_thickness_m_p50_p90', 'grid_predict_s']
out = [{k: r.get(k) for k in keys} | {'MS_check': r.get('MS_iter'), 'OH_check': r.get('OH_iter')} for r in rows]
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'out', 'summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
