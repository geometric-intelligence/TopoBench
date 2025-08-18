---
title: Papers
permalink: /docs/papers/
layout: docs
---

<div class="papers-hero">
    <div class="hero-content">
        <h1>Papers</h1>
        <p class="hero-description">Publications within the TopoBench Ecosystem for Topological Deep Learning</p>
    </div>
</div>

<div class="papers-section">
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

        <div class="paper-card">
            <div class="paper-content">
                <h2>HOPSE: Scalable Higher-Order Positional and Structural Encoder for Combinatorial Representations</h2>
                <p class="authors">Martin Carrasco, Guillermo Bernardez, Marco Montagna, Nina Miolane, Lev Telyatnikov</p>
                <p class="abstract">
                    Existing TDL methods often extend GNNs through Higher-Order Message Passing (HOMP), but face critical scalability challenges due to a combinatorial explosion of message-passing routes and significant complexity overhead from the propagation mechanism. To overcome these limitations, we propose HOPSE (Higher-Order Positional and Structural Encoder) -- a message passing-free framework that uses Hasse graph decompositions to derive efficient and expressive encodings over arbitrary higher-order domains. Notably, HOPSE scales linearly with dataset size while preserving expressive power and permutation equivariance.
                </p>
                <div class="paper-footer">
                    <a href="https://arxiv.org/abs/2505.15405" class="paper-link" target="_blank">
                        <i class="fa fa-external-link"></i> View Paper
                    </a>
                    <span class="paper-date">May 2025</span>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
/* Hero Section - Same as Team page */
.papers-hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background: -webkit-linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background: -moz-linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 4rem 2rem;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.papers-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.1);
    z-index: 1;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
}

.papers-hero h1 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    margin: 0 0 1.5rem 0;
    letter-spacing: -0.03em;
    text-align: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    color: white;
    line-height: 1.2;
}

.hero-description {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 1.5rem;
    line-height: 1.7;
    margin: 0;
    opacity: 0.95;
    font-weight: 400;
}

/* Papers Section */
.papers-section {
    padding: 4rem 2rem;
    max-width: 1200px;
    margin: 0 auto;
    background: #f8fafc;
}

/* Responsive Design for Hero */
@media (max-width: 768px) {
    .papers-hero {
        padding: 3rem 1.5rem;
        min-height: 250px;
    }
    
    .papers-hero h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .hero-description {
        font-size: 1.375rem;
        line-height: 1.6;
    }
    
    .papers-section {
        padding: 3rem 1.5rem;
    }
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
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
    margin: 0 0 15px 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}

.authors {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #64748b;
    font-size: 1.125rem;
    margin-bottom: 20px;
    font-weight: 500;
}

.abstract {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #475569;
    line-height: 1.6;
    margin-bottom: 25px;
    font-size: 1.25rem;
}

.paper-footer {
    border-top: 1px solid rgba(0, 0, 0, 0.08);
    padding-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.paper-link {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    display: inline-flex;
    align-items: center;
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-decoration: none;
    border-radius: 999px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.paper-link:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    color: white;
    text-decoration: none;
}

.paper-link i {
    margin-right: 0.5rem;
}

.paper-date {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #64748b;
    font-size: 1rem;
    font-weight: 500;
}
</style> 