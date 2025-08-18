---
title: Cocitation Citeseer
permalink: /docs/datasets/cocitation-citeseer/
layout: docs
sectionid: docs
---

<div class="dataset-page">
    <div class="dataset-header">
        <h1>Cora</h1>
        <div class="dataset-meta">
            <span class="domain-tag">Graph</span>
            <span class="task-tag">Task Level: Graph</span>
        </div>
    </div>

    <div class="dataset-content">
        <section class="dataset-description">
            <h2>Description</h2>
            <p>The Cora dataset is a citation network dataset consisting of machine learning papers. Each node represents a scientific publication and edges represent citation relationships.</p>
        </section>

        <section class="dataset-statistics">
            <h2>Dataset Overview</h2>
            
            <div class="stats-grid">
                <div class="stats-card main-stats">
                    <h3>Key Numbers</h3>
                    <div class="key-stats">
                        <div class="stat-item">
                            <span class="stat-value">2,708</span>
                            <span class="stat-label">Nodes</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">9,856</span>
                            <span class="stat-label">Total Cells</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">3</span>
                            <span class="stat-label">Max Cell Dimension</span>
                        </div>
                    </div>
                </div>

                <div class="stats-card domain-stats">
                    <h3>Domain Statistics</h3>
                    <div class="statistics-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Domain</th>
                                    <th>0-cell</th>
                                    <th>1-cell</th>
                                    <th>2-cell</th>
                                    <th>3-cell</th>
                                    <th>Hyperedges</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Cellular</td>
                                    <td>2,708</td>
                                    <td>5,278</td>
                                    <td>2,648</td>
                                    <td>0</td>
                                    <td>0</td>
                                </tr>
                                <tr>
                                    <td>Simplicial</td>
                                    <td>2,708</td>
                                    <td>5,278</td>
                                    <td>1,630</td>
                                    <td>220</td>
                                    <td>0</td>
                                </tr>
                                <tr>
                                    <td>Hypergraph</td>
                                    <td>2,708</td>
                                    <td>0</td>
                                    <td>0</td>
                                    <td>0</td>
                                    <td>2,708</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="stats-card lifting-methods">
                    <h3>Lifting Methods</h3>
                    <div class="methods-grid">
                        <div class="method-item">
                            <span class="method-domain">Cellular</span>
                            <span class="method-name">Cycle-based lifting</span>
                        </div>
                        <div class="method-item">
                            <span class="method-domain">Simplicial</span>
                            <span class="method-name">Clique complex lifting</span>
                        </div>
                        <div class="method-item">
                            <span class="method-domain">Hypergraph</span>
                            <span class="method-name">k-hop lifting</span>
                        </div>
                        <div class="method-item">
                            <span class="method-domain">Feature</span>
                            <span class="method-name">Projected sum</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section class="benchmark-section">
            <h2>Model Performance</h2>
            
            <div class="benchmark-container">
                <div class="benchmark-visualization">
                    <div class="benchmark-chart">
                        <canvas id="benchmarkChart"></canvas>
                    </div>
                    
                    <div class="benchmark-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Accuracy (%)</th>
                                    <th>Std Dev (±)</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr class="highlight">
                                    <td>GIN</td>
                                    <td>87.21</td>
                                    <td>1.89</td>
                                </tr>
                                <tr>
                                    <td>GCN</td>
                                    <td>87.09</td>
                                    <td>0.20</td>
                                </tr>
                                <tr>
                                    <td>GAT</td>
                                    <td>86.71</td>
                                    <td>0.95</td>
                                </tr>
                                <tr>
                                    <td>UniGNN2</td>
                                    <td>86.97</td>
                                    <td>0.88</td>
                                </tr>
                                <tr>
                                    <td>EDGNN</td>
                                    <td>87.06</td>
                                    <td>1.09</td>
                                </tr>
                                <tr>
                                    <td>AST</td>
                                    <td>88.92</td>
                                    <td>0.44</td>
                                </tr>
                                <tr>
                                    <td>CWN</td>
                                    <td>86.32</td>
                                    <td>1.38</td>
                                </tr>
                                <tr>
                                    <td>CCCN</td>
                                    <td>87.44</td>
                                    <td>1.28</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="benchmark-insights">
                    <h3>Key Insights</h3>
                    <ul>
                        <li>AST achieves the best performance with 88.92% accuracy</li>
                        <li>Most models perform consistently well, with accuracies above 86%</li>
                        <li>GIN shows relatively high variability (±1.89)</li>
                        <li>GCN shows the most stable results with lowest std dev (±0.20)</li>
                    </ul>
                </div>
            </div>
        </section>
    </div>
</div>

<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ... existing styles ... */
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
            backgroundColor: 'rgba(37, 99, 235, 0.5)',
            borderColor: 'rgb(37, 99, 235)',
            borderWidth: 2
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
                    font: {
                        size: 20,
                        weight: '600',
                        family: "'Inter', sans-serif"
                    },
                    padding: 20
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
                        font: {
                            size: 16,
                            weight: '600',
                            family: "'Inter', sans-serif"
                        }
                    },
                    ticks: {
                        font: {
                            size: 14,
                            family: "'Inter', sans-serif"
                        }
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 14,
                            family: "'Inter', sans-serif"
                        }
                    }
                }
            }
        }
    };

    new Chart(ctx, config);
});
</script> 