---
title: Core Concepts
permalink: /docs/documentation/core-concepts/
layout: docs
---

<div class="page-container">
    <div class="documentation-content">
        <h1>Core Concepts</h1>

        <div class="intro-text">
            <p>This section covers the fundamental concepts and principles behind TopoBench, a library designed for benchmarking and research in Topological Deep Learning.</p>
        </div>

        <div class="content-section">
            <h2>Topological Deep Learning</h2>
            <p>Topological Deep Learning (TDL) extends traditional deep learning approaches to work with data that has a topological structure. This includes:</p>
            <ul>
                <li>Graph Neural Networks (GNNs)</li>
                <li>Hypergraph Neural Networks</li>
                <li>Simplicial Neural Networks</li>
                <li>Cell Complex Neural Networks</li>
            </ul>
        </div>

        <div class="content-section">
            <h2>Data Structures</h2>
            <p>TopoBench supports various topological data structures:</p>
            
            <h3>Graphs</h3>
            <p>Traditional graph structures with nodes and edges.</p>
            <div class="code-block">
                <pre><code>from topobench.datasets.graph import GraphDataset
dataset = GraphDataset(name="Cora")</code></pre>
            </div>

            <h3>Hypergraphs</h3>
            <p>Generalized graphs where edges can connect any number of nodes.</p>
            <div class="code-block">
                <pre><code>from topobench.datasets.hypergraph import HypergraphDataset
dataset = HypergraphDataset(name="Cora-Coauthorship")</code></pre>
            </div>

            <h3>Simplicial Complexes</h3>
            <p>Higher-order structures that generalize graphs to include higher-dimensional simplices.</p>
            <div class="code-block">
                <pre><code>from topobench.datasets.simplicial import SimplicialDataset
dataset = SimplicialDataset(name="Mantra")</code></pre>
            </div>
        </div>

        <div class="content-section">
            <h2>Model Architecture</h2>
            <p>TopoBench provides a flexible architecture for building topological neural networks:</p>
            
            <h3>Message Passing</h3>
            <p>The core of topological neural networks is message passing between different elements of the structure.</p>
            <div class="code-block">
                <pre><code>from topobench.models import TopologicalMessagePassing
model = TopologicalMessagePassing(
    in_channels=64,
    hidden_channels=128,
    out_channels=32
)</code></pre>
            </div>

            <h3>Feature Lifting</h3>
            <p>Transforming features between different topological domains.</p>
            <div class="code-block">
                <pre><code>from topobench.transforms import FeatureLifting
lifting = FeatureLifting(
    source_domain="graph",
    target_domain="simplicial"
)</code></pre>
            </div>
        </div>

        <div class="content-section">
            <h2>Benchmarking</h2>
            <p>TopoBench provides standardized benchmarking tools for evaluating models:</p>
            
            <h3>Evaluation Metrics</h3>
            <ul>
                <li>Classification accuracy</li>
                <li>Node-level metrics</li>
                <li>Graph-level metrics</li>
                <li>Topological metrics</li>
            </ul>

            <h3>Benchmarking Pipeline</h3>
            <div class="code-block">
                <pre><code>from topobench.benchmark import Benchmark
benchmark = Benchmark(
    dataset="Cora",
    model="GCN",
    metrics=["accuracy", "f1_score"]
)
results = benchmark.run()</code></pre>
            </div>
        </div>
    </div>
</div>

<style>
.page-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.documentation-content {
    background: #ffffff;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 1.5rem;
}

.intro-text {
    max-width: 800px;
    margin-bottom: 3rem;
}

.intro-text p {
    font-size: 1.125rem;
    line-height: 1.6;
    color: #4a5568;
}

.content-section {
    margin-bottom: 3rem;
}

.content-section h2 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 1.5rem;
}

.content-section h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 1.5rem 0 1rem;
}

.content-section p {
    color: #4a5568;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.content-section ul {
    list-style-type: disc;
    padding-left: 1.5rem;
    margin-bottom: 1rem;
}

.content-section li {
    color: #4a5568;
    margin-bottom: 0.5rem;
}

.code-block {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    overflow-x: auto;
}

.code-block pre {
    margin: 0;
    font-family: 'Fira Code', monospace;
    font-size: 0.875rem;
    line-height: 1.5;
}

.code-block code {
    color: #1a1a1a;
}

@media (max-width: 768px) {
    .page-container {
        padding: 1rem;
    }

    .documentation-content {
        padding: 1.5rem;
    }

    h1 {
        font-size: 2rem;
    }

    .content-section h2 {
        font-size: 1.5rem;
    }

    .content-section h3 {
        font-size: 1.25rem;
    }
}
</style> 