import json
import os
import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = "projects/Optimization/reports"
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
SUMMARY_FILE = os.path.join(REPORTS_DIR, "SUMMARY.md")

def load_json(filename):
    with open(os.path.join(REPORTS_DIR, filename), 'r') as f:
        return json.load(f)

def create_summary_md(reports):
    with open(SUMMARY_FILE, 'w') as f:
        f.write("# Optimization Reports Summary\n\n")
        
        for name, data in reports.items():
            f.write(f"## {data.get('project', name)}\n\n")
            
            if "baseline" in data and ("compressed" in data or "quantized" in data):
                target_key = "compressed" if "compressed" in data else "quantized"
                target_data = data[target_key]
                
                f.write("| Metric | Baseline | " + target_key.capitalize() + " | Delta/Ratio |\n")
                f.write("| --- | --- | --- | --- |\n")
                
                # Accuracy
                base_acc = data['baseline'].get('evaluation', {}).get('accuracy', data['baseline'].get('accuracy', 'N/A'))
                target_acc = target_data.get('evaluation', {}).get('accuracy', target_data.get('accuracy', 'N/A'))
                acc_delta = data.get('comparison', {}).get('accuracy_delta', 'N/A')
                f.write(f"| Accuracy | {base_acc} | {target_acc} | {acc_delta} |\n")
                
                # Latency
                base_lat = data['baseline'].get('profile', {}).get('latency_ms', data['baseline'].get('latency_ms', 'N/A'))
                target_lat = target_data.get('profile', {}).get('latency_ms', target_data.get('latency_ms', 'N/A'))
                lat_delta = data.get('comparison', {}).get('latency_delta_ms', 'N/A')
                f.write(f"| Latency (ms) | {base_lat} | {target_lat} | {lat_delta} |\n")
                
                # Memory/Size
                base_mem = data['baseline'].get('profile', {}).get('peak_memory_mb', data['baseline'].get('estimated_model_bytes', 'N/A'))
                target_mem = target_data.get('profile', {}).get('peak_memory_mb', target_data.get('estimated_model_bytes', 'N/A'))
                f.write(f"| Memory/Size | {base_mem} | {target_mem} | - |\n")
                
            elif name == "kv_cache_benchmark.json":
                f.write(f"- Naive Latency: {data['naive_latency_ms']:.2f} ms\n")
                f.write(f"- Cached Latency: {data['cached_latency_ms']:.2f} ms\n")
                f.write(f"- Speedup: {data['speedup_x']:.2f}x\n")
            
            plot_path = os.path.join("plots", f"{name.replace('.json', '')}.png")
            if os.path.exists(os.path.join(REPORTS_DIR, plot_path)):
                f.write(f"\n![{name}]({plot_path})\n")
            
            f.write("\n---\n\n")

def plot_comparison(name, data):
    if "baseline" in data and ("compressed" in data or "quantized" in data):
        target_key = "compressed" if "compressed" in data else "quantized"
        target_data = data[target_key]
        
        metrics = ['Accuracy', 'Latency (ms)']
        baseline_vals = [
            data['baseline'].get('evaluation', {}).get('accuracy', data['baseline'].get('accuracy', 0)),
            data['baseline'].get('profile', {}).get('latency_ms', data['baseline'].get('latency_ms', 0))
        ]
        target_vals = [
            target_data.get('evaluation', {}).get('accuracy', target_data.get('accuracy', 0)),
            target_data.get('profile', {}).get('latency_ms', target_data.get('latency_ms', 0))
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, baseline_vals, width, label='Baseline')
        ax.bar(x + width/2, target_vals, width, label=target_key.capitalize())
        
        ax.set_ylabel('Value')
        ax.set_title(f'Baseline vs {target_key.capitalize()}: {data.get("project", name)}')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{name.replace('.json', '')}.png"))
        plt.close()

def plot_kv_cache(name, data):
    labels = ['Naive', 'KV-Cached']
    latencies = [data['naive_latency_ms'], data['cached_latency_ms']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, latencies, color=['red', 'green'])
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'KV Cache Benchmark (Speedup: {data["speedup_x"]:.2f}x)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name.replace('.json', '')}.png"))
    plt.close()

def main():
    files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
    reports = {}
    for f in files:
        data = load_json(f)
        reports[f] = data
        if "kv_cache" in f:
            plot_kv_cache(f, data)
        else:
            plot_comparison(f, data)
    
    create_summary_md(reports)
    print(f"Visualization complete. Summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
