---
title: Papers
permalink: /docs/papers/
layout: docs
---

<div class="papers-header" style="background: linear-gradient(135deg, #2196F3, #00BCD4); padding: 60px 0; margin: -20px -15px 40px -15px; width: 100vw; position: relative; left: 50%; right: 50%; margin-left: -50vw; margin-right: -50vw;">
    <div class="container text-center">
        <h1 style="color: white; font-size: 48px; margin-bottom: 20px;">Papers</h1>
        <p style="color: white; font-size: 20px;">Research publications in Topological Deep Learning</p>
    </div>
</div>

<div class="container">
    <div class="papers-container">
        <div class="paper-card">
            <div class="paper-content">
                <h2>TopoBench: A Framework for Benchmarking Topological Deep Learning</h2>
                <p class="authors">Lev Telyatnikov, Guillermo Bernardez, Marco Montagna, et al., Nina Miolane, Simone Scardapane, Theodore Papamarkou</p>
                <p class="abstract">
                    This work introduces TopoBench, an open-source library designed to standardize benchmarking and accelerate research in topological deep learning (TDL). TopoBench decomposes TDL into a sequence of independent modules for data generation, loading, transforming and processing, as well as model training, optimization and evaluation...
                </p>
                <div class="paper-footer">
                    <a href="https://arxiv.org/abs/2406.06642" class="paper-link" target="_blank">
                        <i class="fa fa-external-link"></i> View Paper
                    </a>
                    <span class="paper-date">Jun 2024</span>
                </div>
            </div>
        </div>

        <div class="paper-card">
            <div class="paper-content">
                <h2>TopoTune: A Framework for Generalized Combinatorial Complex Neural Networks</h2>
                <p class="authors">Mathilde Papillon, Guillermo Bernárdez, Claudio Battiloro, Nina Miolane</p>
                <p class="abstract">
                    Graph Neural Networks (GNNs) excel in learning from relational datasets, processing node and edge features in a way that preserves the symmetries of the graph domain. However, many complex systems -- such as biological or social networks--involve multiway complex interactions that are more naturally represented by higher-order topological domains. The emerging field of Topological Deep Learning (TDL) aims to accommodate and leverage these higher-order structures...
                </p>
                <div class="paper-footer">
                    <a href="https://arxiv.org/abs/2410.06530" class="paper-link" target="_blank">
                        <i class="fa fa-external-link"></i> View Paper
                    </a>
                    <span class="paper-date">Oct 2024</span>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
.papers-header {
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.papers-container {
    max-width: 900px;
    margin: 0 auto;
}

.paper-card {
    background: white;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}

.paper-card:hover {
    transform: translateY(-5px);
}

.paper-content {
    padding: 30px;
}

.paper-content h2 {
    color: #333;
    margin: 0 0 15px 0;
    font-size: 24px;
    font-weight: 600;
}

.authors {
    color: #666;
    font-size: 16px;
    margin-bottom: 20px;
}

.abstract {
    color: #444;
    line-height: 1.6;
    margin-bottom: 25px;
}

.paper-footer {
    border-top: 1px solid #eee;
    padding-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.paper-link {
    display: inline-flex;
    align-items: center;
    padding: 10px 20px;
    background: #2196F3;
    color: white;
    text-decoration: none;
    border-radius: 25px;
    transition: background 0.2s;
}

.paper-link:hover {
    background: #1976D2;
    color: white;
    text-decoration: none;
}

.paper-link i {
    margin-right: 8px;
}

.paper-date {
    color: #666;
    font-size: 14px;
}
</style> 