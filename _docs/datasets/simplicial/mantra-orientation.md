---
title: Mantra Orientation
permalink: /docs/datasets/simplicial/mantra-orientation/
layout: docs
sectionid: docs
---

<div class="dataset-page clean-modern v2">
    <div class="dataset-header clean-header v2-header">
        <h1>Mantra Orientation</h1>
        <div class="dataset-meta v2-meta">
            <span class="domain-tag">Graph</span>
            <span class="task-tag">Task Level: Graph</span>
        </div>
    </div>

    <div class="dataset-content clean-content v2-content">
        <section class="description-card v2-card">
            <h2>Description</h2>
            <p>The Mantra Orientation dataset is a citation network dataset consisting of scientific publications from Mantra Orientation. Each node represents a scientific publication and edges represent citation relationships.</p>
        </section>

        <section class="overview-card v2-card">
            <h2>Dataset Overview</h2>
            <div class="overview-grid v2-grid">
                <div class="overview-block key-numbers v2-block">
                    <h3>Key Numbers</h3>
                    <div class="key-stats v2-key-stats" style="gap:0.7rem;">
                        <div class="stat-item"><span class="stat-value blue v2-big">2,708</span><span class="stat-label">Nodes</span></div>
                        <div class="stat-item"><span class="stat-value blue v2-big">9,856</span><span class="stat-label">Total Cells</span></div>
                        <div class="stat-item"><span class="stat-value blue v2-big">3</span><span class="stat-label">Max Cell Dimension</span></div>
                    </div>
                </div>

                <div class="overview-block domain-stats v2-block">
                    <h3>Domain Statistics</h3>
                    <div class="statistics-table v2-table large-table">
                        <table>
                            <thead>
                                <tr><th>Domain</th><th>0-cell</th><th>1-cell</th><th>2-cell</th><th>3-cell</th><th>Hyperedges</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Cellular</td><td>2,708</td><td>5,278</td><td>2,648</td><td>0</td><td>0</td></tr>
                                <tr><td>Simplicial</td><td>2,708</td><td>5,278</td><td>1,630</td><td>220</td><td>0</td></tr>
                                <tr><td>Hypergraph</td><td>2,708</td><td>0</td><td>0</td><td>0</td><td>2,708</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="overview-block lifting-methods v2-block">
                    <h3>Lifting Methods</h3>
                    <div class="modern-lifting">
                        <div class="lifting-group">
                            <div class="lifting-title">Structural-based Liftings</div>
                            <ul>
                                <li><b>Cellular:</b> <a href="https://github.com/geometric-intelligence/TopoBench/wiki/Cycle-Lifting-(Graph-to-Cell)" target="_blank" rel="noopener">Cycle-based lifting</a></li>
                                <li><b>Simplicial:</b> <a href="https://github.com/geometric-intelligence/TopoBench/wiki/Clique-Lifting-(Graph-to-Simplicial)" target="_blank" rel="noopener">Clique complex lifting</a></li>
                                <li><b>Hypergraph:</b> <a href="https://github.com/geometric-intelligence/TopoBench/wiki/KHop-Lifting-(Graph-to-Hypergraph)" target="_blank" rel="noopener">k-hop lifting</a></li>
                            </ul>
                        </div>
                        <div class="lifting-divider"></div>
                        <div class="lifting-group">
                            <div class="lifting-title">Feature Lifting</div>
                            <ul>
                                <li><b>Projected sum</b></li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section class="performance-card v2-card">
            <h2>Model Performance</h2>
            
            <div class="performance-grid v2-perf-grid">
                <div class="performance-chart v2-perf-chart">
                    <canvas id="benchmarkChart" height="120"></canvas>
                </div>
                
                <div class="performance-table-wrapper v2-table-wrapper">
                    <table class="performance-table v2-perf-table">
                        <thead>
                            <tr><th>Model</th><th>Accuracy (%)</th><th>Std Dev (±)</th></tr>
                        </thead>
                        <tbody>
                            <tr class="highlight"><td>GIN</td><td>87.21</td><td>1.89</td></tr>
                            <tr><td>GCN</td><td>87.09</td><td>0.20</td></tr>
                            <tr><td>GAT</td><td>86.71</td><td>0.95</td></tr>
                            <tr><td>UniGNN2</td><td>86.97</td><td>0.88</td></tr>
                            <tr><td>EDGNN</td><td>87.06</td><td>1.09</td></tr>
                            <tr><td>AST</td><td>88.92</td><td>0.44</td></tr>
                            <tr><td>CWN</td><td>86.32</td><td>1.38</td></tr>
                            <tr><td>CCCN</td><td>87.44</td><td>1.28</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="insights-card v2-insights">
                <h3>Key Insights</h3>
                <ul>
                    <li>AST achieves the best performance with 88.92% accuracy</li>
                    <li>Most models perform consistently well, with accuracies above 86%</li>
                    <li>GIN shows relatively high variability (±1.89)</li>
                    <li>GCN shows the most stable results with lowest std dev (±0.20)</li>
                </ul>
                <div class="insight-divider"></div>
                <div class="repro-block">
                    <div class="repro-title-copy-row">
                        <div class="repro-title">Reproducibility</div>
                        <button class="copy-btn" onclick="copyReproCmd(this)" title="Copy to clipboard">
                            <svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="7" width="9" height="9" rx="2" stroke="#2563eb" stroke-width="1.5"/><rect x="7.5" y="4" width="9" height="9" rx="2" fill="#fff" stroke="#2563eb" stroke-width="1.5"/></svg>
                        </button>
                    </div>
                    <pre class="repro-cli" id="repro-cli">python -m topobench model=graph/gin dataset=graph/cocitation_Mantra Orientation optimizer.parameters.lr=0.001 model.feature_encoder.out_channels=64 model.backbone.num_layers=2 model.feature_encoder.proj_dropout=0.5 dataset.dataloader_params.batch_size=1 dataset.split_params.data_seed=0,3,5,7,9 trainer.max_epochs=500 trainer.min_epochs=50 trainer.check_val_every_n_epoch=1 callbacks.early_stopping.patience=50</pre>
                </div>
            </div>
        </section>
    </div>
</div>

<style>
body {
    background: #f6faff;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.dataset-page.clean-modern.v2 {
    max-width: 1100px;
    margin: 0 auto;
    padding: 3.5rem 1.5rem 4rem 1.5rem;
}
.clean-header.v2-header {
    text-align: center;
    margin-bottom: 2.8rem;
    margin-top: 1.5rem;
}
.clean-header.v2-header h1 {
    font-size: 3.1rem;
    font-weight: 900;
    color: #22223b;
    margin-bottom: 0.5rem;
    letter-spacing: -0.04em;
}
.dataset-meta.v2-meta {
    display: flex;
    justify-content: center;
    gap: 0.7rem;
    margin-top: 0.1rem;
}
.domain-tag, .task-tag {
    font-size: 0.98rem;
    padding: 0.22rem 1.1rem;
    border-radius: 999px;
    font-weight: 700;
}
.domain-tag {
    background: #e3edff;
    color: #2563eb;
}
.task-tag {
    background: #e6fbe8;
    color: #16a34a;
}
.clean-content.v2-content {
    display: flex;
    flex-direction: column;
    gap: 2.8rem;
}
.v2-card {
    background: #fcfdff;
    border-radius: 22px;
    box-shadow: 0 4px 32px rgba(37,99,235,0.09);
    padding: 2.3rem 2.7rem;
    margin-bottom: 0;
}
.description-card h2 {
    font-size: 1.25rem;
    font-weight: 800;
    color: #22223b;
    margin-bottom: 0.7rem;
}
.description-card p {
    font-size: 1.13rem;
    color: #475569;
    margin: 0;
}
.overview-card h2, .performance-card h2 {
    font-size: 1.6rem;
    font-weight: 900;
    color: #22223b;
    margin-bottom: 2.1rem;
}
.overview-grid.v2-grid {
    display: grid;
    grid-template-columns: 1.1fr 1.6fr 0.8fr;
    gap: 1.1rem;
    align-items: flex-start;
}
.overview-block h3 {
    font-size: 1.08rem;
    font-weight: 700;
    color: #22223b;
    margin-bottom: 1.1rem;
}
.key-stats.v2-key-stats {
    display: flex;
    flex-direction: column;
    gap: 0.7rem !important;
    margin-top: 0.5rem;
}
.stat-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.1rem;
}
.stat-value {
    font-size: 2.5rem;
    font-weight: 900;
    color: #2563eb;
    letter-spacing: -0.01em;
}
.stat-value.v2-big {
    font-size: 2.7rem;
    font-weight: 900;
}
.stat-value.blue {
    color: #2563eb;
}
.stat-label {
    font-size: 1.08rem;
    color: #64748b;
}
.statistics-table.v2-table.large-table table {
    font-size: 1.18rem;
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #f8fafc;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(37,99,235,0.04);
    margin-top: 0;
}
.statistics-table.v2-table.large-table th, .statistics-table.v2-table.large-table td {
    padding: 0.7rem 1.1rem;
    text-align: center;
    transition: background 0.2s;
}
.statistics-table.v2-table.large-table th {
    background: #e3edff;
    color: #2563eb;
    font-weight: 800;
}
.statistics-table.v2-table.large-table tr.highlight {
    background: #e0e7ff;
    font-weight: 700;
}
.statistics-table.v2-table.large-table tr:nth-child(even) {
    background: #f1f5f9;
}
.statistics-table.v2-table.large-table tr:hover {
    background: #e0e7ff;
}

/* Modern Lifting Methods */
.modern-lifting {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    background: #f8fafc;
    border-radius: 10px;
    padding: 1.1rem 1.2rem 1.1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(37,99,235,0.04);
}
.lifting-group {
    margin-bottom: 0.2rem;
}
.lifting-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #2563eb;
    margin-bottom: 0.3rem;
    letter-spacing: 0.01em;
}
.lifting-group ul {
    margin: 0 0 0 0.7rem;
    padding: 0;
    list-style: disc inside;
    font-size: 1.05rem;
    color: #334155;
}
.lifting-divider {
    border-top: 1.5px solid #e3edff;
    margin: 0.5rem 0 0.5rem 0;
}
.performance-card h2 {
    font-size: 1.6rem;
    font-weight: 900;
    color: #22223b;
    margin-bottom: 2.1rem;
}
.performance-grid.v2-perf-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 2.1rem;
    align-items: flex-start;
}
.performance-chart.v2-perf-chart {
    background: #f8fafc;
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    box-shadow: 0 2px 8px rgba(37,99,235,0.06);
    display: flex;
    align-items: center;
    justify-content: center;
}
.performance-table-wrapper.v2-table-wrapper {
    background: #f8fafc;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.06);
    padding: 1.2rem 1rem 1rem 1rem;
    margin-top: 0;
}
.performance-table.v2-perf-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 1.08rem;
    margin-top: 0;
    box-shadow: 0 1px 4px rgba(37,99,235,0.04);
    border-radius: 10px;
    overflow: hidden;
}
.performance-table.v2-perf-table th, .performance-table.v2-perf-table td {
    padding: 0.4rem 0.7rem;
    text-align: center;
    transition: background 0.2s;
}
.performance-table.v2-perf-table th {
    background: #e3edff;
    color: #2563eb;
    font-weight: 800;
}
.performance-table.v2-perf-table tr.highlight {
    background: #e0e7ff;
    font-weight: 700;
}
.performance-table.v2-perf-table tr:nth-child(even) {
    background: #f1f5f9;
}
.performance-table.v2-perf-table tr:hover {
    background: #e0e7ff;
}
.insights-card.v2-insights {
    background: #f8fafc;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.06);
    padding: 1.5rem 1.2rem 1.2rem 1.2rem;
    margin-top: 2.1rem;
    color: #1e293b;
    border-left: 5px solid #2563eb;
}
.insights-card.v2-insights h3 {
    font-size: 1.15rem;
    font-weight: 800;
    color: #2563eb;
    margin-bottom: 0.7rem;
}
.insights-card.v2-insights ul {
    padding-left: 1.2rem;
    margin: 0;
    font-size: 1.08rem;
}
.overview-block.lifting-methods.v2-block {
    margin-left: 0.5rem;
}
.overview-block.domain-stats.v2-block {
    margin-left: -3cm;
}
.insight-divider {
    border-top: 1.5px solid #e3edff;
    margin: 1.2rem 0 1.2rem 0;
}
.repro-block {
    margin-top: 0.2rem;
}
.repro-title-copy-row {
    display: flex;
    align-items: flex-end;
    justify-content: flex-end;
    margin-bottom: -2.1rem;
    margin-right: 0.7rem;
    position: relative;
    z-index: 2;
}
.repro-title {
    display: none;
}
.copy-btn {
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.1rem 0.2rem;
    border-radius: 5px;
    transition: background 0.15s;
    margin-left: 0;
    margin-right: 0.2rem;
    position: relative;
    top: 0.1rem;
}
.copy-btn:hover {
    background: #e3edff;
}
.copy-btn.copied svg rect {
    stroke: #16a34a;
}
.copy-btn:active {
    background: #dbeafe;
}
@media (max-width: 1100px) {
    .overview-grid.v2-grid, .performance-grid.v2-perf-grid {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
    .clean-header.v2-header {
        padding: 1.2rem 1rem 1rem 1rem;
    }
    .v2-card {
        padding: 1.2rem 1rem;
    }
}
</style>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('benchmarkChart').getContext('2d');
    
    const data = {
        labels: ['AST', 'GIN', 'GCN', 'CCCN', 'EDGNN', 'UniGNN2', 'GAT', 'CWN'],
        datasets: [{
            label: 'Accuracy (%)',
            data: [88.92, 87.21, 87.09, 87.44, 87.06, 86.97, 86.71, 86.32],
            backgroundColor: 'rgba(37, 99, 235, 0.12)',
            borderColor: 'rgb(37, 99, 235)',
            borderWidth: 2,
            borderRadius: 10,
            barPercentage: 0.7,
            categoryPercentage: 0.7
        }]
    };

    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Model Performance Comparison',
                    font: { size: 18, weight: '700', family: "'Inter', sans-serif" },
                    color: '#22223b',
                    padding: 18
                },
                tooltip: {
                    backgroundColor: '#2563eb',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#e3edff',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 85,
                    max: 90,
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        font: { size: 16, weight: '700', family: "'Inter', sans-serif" }
                    },
                    ticks: {
                        font: { size: 15, family: "'Inter', sans-serif" },
                        color: '#64748b',
                        stepSize: 1
                    },
                    grid: {
                        color: '#e3edff',
                        borderDash: [4, 4]
                    }
                },
                x: {
                    ticks: {
                        font: { size: 15, family: "'Inter', sans-serif" },
                        color: '#64748b'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    };

    new Chart(ctx, config);
});

function copyReproCmd(btn) {
    const code = document.getElementById('repro-cli').innerText;
    navigator.clipboard.writeText(code);
    btn.classList.add('copied');
    btn.title = 'Copied!';
    setTimeout(() => {
        btn.classList.remove('copied');
        btn.title = 'Copy to clipboard';
    }, 1200);
}
</script> 