---
title: Getting Started
permalink: /docs/getting-started/
---

<div class="getting-started-hero">
    <div class="hero-content">
        <h1>Getting Started with TopoBench</h1>
        <p class="hero-description">
            TopoBench is a modular Python library for benchmarking and accelerating research in Topological Deep Learning (TDL). It enables training and comparison of Topological Neural Networks (TNNs) across graphs, simplicial complexes, cellular complexes, and hypergraphs.
        </p>
    </div>
</div>

<div class="getting-started-section">
    <div class="workflow-container">
        <div class="workflow-image">
            <img src="{{ site.baseurl }}/assets/img/workflow.jpg" alt="TopoBench Workflow">
        </div>
    </div>

    <div class="info-card">
        <div class="card-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="card-content">
            <h3>Technical Details</h3>
            <p>For comprehensive information, see our research paper: <a href="https://arxiv.org/pdf/2406.06642" target="_blank" class="paper-link">TopoBench: A Framework for Benchmarking Topological Deep Learning</a></p>
        </div>
    </div>

    <div class="setup-section">
        <h2>1. Environment Setup</h2>
        <div class="requirements-box">
            <strong>Requirements:</strong> Python 3.11.3, <a href="https://docs.conda.io/en/latest/" target="_blank">conda</a> (or Miniconda)
        </div>
        
        <ol class="setup-steps">
            <li>
                <span class="step-number">1</span>
                <div class="step-content">
                    <strong>Check conda installation:</strong>
                    <pre><code>conda --version</code></pre>
                    <p class="step-note">If not installed, get Miniconda <a href="https://www.anaconda.com/docs/getting-started/miniconda/install" target="_blank">here</a>.</p>
                </div>
            </li>
            <li>
                <span class="step-number">2</span>
                <div class="step-content">
                    <strong>Clone the repository and enter the directory:</strong>
                    <pre><code>git clone git@github.com:geometric-intelligence/topobench.git
cd TopoBench</code></pre>
                </div>
            </li>
            <li>
                <span class="step-number">3</span>
                <div class="step-content">
                    <strong>Create and activate the environment:</strong>
                    <pre><code>conda create -n tb python=3.11.3
conda activate tb</code></pre>
                </div>
            </li>
            <li>
                <span class="step-number">4</span>
                <div class="step-content">
                    <strong>(Optional, for GPU) Check CUDA version:</strong>
                    <pre><code>which nvcc && nvcc --version</code></pre>
                    <p class="step-note">Ensure CUDA matches <code>env_setup.sh</code> (<code>CUDA=cpu</code> by default). Adjust <code>CUDA</code> and <code>TORCH</code> as needed. See <a href="https://github.com/pyg-team/pyg-lib" target="_blank">compatibility guide</a>.</p>
                </div>
            </li>
            <li>
                <span class="step-number">5</span>
                <div class="step-content">
                    <strong>Install dependencies:</strong>
                    <pre><code>source env_setup.sh</code></pre>
                </div>
            </li>
        </ol>
    </div>

    <div class="training-section">
        <h2>2. Running the Training Pipeline</h2>
        <p>After setup, train and evaluate a neural network with:</p>
        <pre class="code-block"><code>python -m topobench</code></pre>
    </div>

    <div class="customization-section">
        <h2>3. Customizing Experiments</h2>
        <p><strong>TopoBench</strong> uses <strong>hydra</strong> for flexible configuration. Override defaults via the command line:</p>
        <pre class="code-block"><code>python -m topobench model=cell/cwn dataset=graph/MUTAG</code></pre>
        
        <div class="info-card note-card">
            <div class="card-icon note-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <path d="M12 16V12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 8H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="card-content">
                <h3>Note</h3>
                <p>By default, the pipeline detects source/destination domains and applies a default lifting if needed.</p>
            </div>
        </div>

        <div class="config-section">
            <h3>Configuring Individual Transforms</h3>
            <ol class="config-steps">
                <li>Select a transform (e.g., a lifting transform).</li>
                <li>Find its config path (see below):
                    <pre class="code-block"><code>├── configs
│   ├── data_manipulations
│   ├── transforms
│   │   └── liftings
│   │       ├── graph2cell
│   │       ├── graph2hypergraph
│   │       └── graph2simplicial</code></pre>
                </li>
                <li>Override the default transform:
                    <pre class="code-block"><code>python -m topobench model=&lt;model_type&gt;/&lt;model_name&gt; dataset=&lt;data_type&gt;/&lt;dataset_name&gt; transforms=[&lt;transform_path&gt;/&lt;transform_name&gt;]</code></pre>
                </li>
                <li>Example:
                    <pre class="code-block"><code>python -m topobench model=cell/cwn dataset=graph/MUTAG transforms=[liftings/graph2cell/discrete_configuration_complex]</code></pre>
                </li>
            </ol>
        </div>

        <div class="config-section">
            <h3>Configuring Transform Groups</h3>
            <ol class="config-steps">
                <li>Create a config file in <code>configs/transforms</code> (e.g., <code>custom_example.yaml</code>).</li>
                <li>Define the group in YAML:
                    <pre class="code-block"><code>defaults:
- data_manipulations@data_transform_1: identity
- data_manipulations@data_transform_2: node_degrees
- data_manipulations@data_transform_3: one_hot_node_degree_features
- liftings/graph2cell@graph2cell_lifting: cycle</code></pre>
                </li>
                <li>Run with the custom group:
                    <pre class="code-block"><code>python -m topobench model=cell/cwn dataset=graph/ZINC transforms=custom_example</code></pre>
                </li>
            </ol>
        </div>

        <div class="info-card tip-card">
            <div class="card-icon tip-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <path d="M12 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="card-content">
                <h3>Tip</h3>
                <p>Use <a href="https://hydra.cc/docs/intro/" target="_blank">hydra documentation</a> for advanced configuration.</p>
            </div>
        </div>
    </div>

    <div class="reproducibility-section">
        <h2>4. Reproducibility</h2>
        <p>To reproduce Table 1 from the TopoBench paper:</p>
        <pre class="code-block"><code>bash scripts/reproduce.sh</code></pre>
        
        <div class="info-card info-card-blue">
            <div class="card-icon info-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                    <path d="M12 16V12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <path d="M12 8H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
            </div>
            <div class="card-content">
                <h3>Info</h3>
                <p>Public <a href="https://wandb.ai/telyatnikov_sap/TopoBenchmark_main?nw=nwusertelyatnikov_sap" target="_blank">W&B (Weights & Biases) project</a> contains logs for all runs (updated June 11, 2024).</p>
            </div>
        </div>
    </div>

    <div class="resources-section">
        <h2>5. Tutorials & Further Resources</h2>
        <ul class="resources-list">
            <li>See <a href="https://github.com/geometric-intelligence/TopoBench/tree/main/tutorials" target="_blank">tutorials</a> for adding datasets, transforms, and benchmarks.</li>
            <li>Many neural networks in TopoBench are from <a href="https://github.com/pyt-team/TopoModelX" target="_blank">TopoModelX</a>.</li>
            <li>For questions, open an issue on <a href="https://github.com/geometric-intelligence/TopoBench" target="_blank">GitHub</a>.</li>
        </ul>
    </div>
</div>

<style>
/* Modern Professional Styling - Same as Team and Papers pages */
body {
    background: #f8fafc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}

/* Hero Section - Same as Team and Papers pages */
.getting-started-hero {
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

.getting-started-hero::before {
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

.getting-started-hero h1 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 3rem;
    font-weight: 800;
    margin: 2rem 0 1.5rem 0;
    letter-spacing: -0.03em;
    text-align: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    color: white;
    line-height: 1.2;
}

.hero-description {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 2rem;
    line-height: 1.7;
    margin: 0;
    opacity: 0.95;
    font-weight: 400;
}

/* Getting Started Section */
.getting-started-section {
    padding: 4rem 2rem;
    max-width: 1200px;
    margin: 0 auto;
    background: #f8fafc;
}

/* Workflow Image */
.workflow-container {
    text-align: center;
    margin-bottom: 3rem;
}

.workflow-image img {
    max-width: 700px;
    width: 100%;
    border-radius: 16px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.workflow-image img:hover {
    transform: translateY(-4px);
}

/* Info Cards */
.info-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    display: flex;
    align-items: flex-start;
    gap: 1.5rem;
    border-left: 4px solid #667eea;
}

.card-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    flex-shrink: 0;
}

.note-icon {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}

.tip-icon {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.info-icon {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
}

.card-content h3 {
    font-size: 1.875rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 0.75rem 0;
}

.card-content p {
    font-size: 1.5625rem;
    color: #475569;
    line-height: 1.6;
    margin: 0;
}

.paper-link {
    color: #667eea;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s ease;
}

.paper-link:hover {
    color: #5a67d8;
    text-decoration: underline;
}

/* Section Headers */
.getting-started-section h2 {
    font-size: 3.125rem;
    font-weight: 700;
    color: #1e293b;
    margin: 3rem 0 1.5rem 0;
    letter-spacing: -0.01em;
}

.getting-started-section h3 {
    font-size: 2.1875rem;
    font-weight: 600;
    color: #1e293b;
    margin: 2rem 0 1rem 0;
}

/* Requirements Box */
.requirements-box {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 1.5625rem;
    color: #475569;
}

.requirements-box a {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
}

.requirements-box a:hover {
    text-decoration: underline;
}

/* Setup Steps */
.setup-steps {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.setup-steps li {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0;
}

.step-number {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.875rem;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.step-content {
    flex: 1;
}

.step-content strong {
    display: block;
    font-size: 1.5625rem;
    color: #1e293b;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.step-content pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.75rem 0;
    overflow-x: auto;
}

.step-content code {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 0.9375rem;
    color: #1e293b;
}

.step-note {
    font-size: 0.9375rem;
    color: #64748b;
    margin: 0.5rem 0 0 0;
    line-height: 1.5;
}

.step-note a {
    color: #667eea;
    text-decoration: none;
}

.step-note a:hover {
    text-decoration: underline;
}

/* Code Blocks */
.code-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    overflow-x: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.code-block code {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 1.40625rem;
    color: #1e293b;
    line-height: 1.5;
}

/* Config Sections */
.config-section {
    margin: 2rem 0;
}

.config-steps {
    margin: 1rem 0;
    padding-left: 1.5rem;
}

.config-steps li {
    margin-bottom: 1rem;
    font-size: 1.5625rem;
    color: #475569;
    line-height: 1.6;
}

.config-steps code {
    background: #f1f5f9;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    color: #1e293b;
}

/* Resources List */
.resources-list {
    margin: 1rem 0;
    padding-left: 1.5rem;
}

.resources-list li {
    margin-bottom: 0.75rem;
    font-size: 1.5625rem;
    color: #475569;
    line-height: 1.6;
}

.resources-list a {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
}

.resources-list a:hover {
    text-decoration: underline;
}

/* Responsive Design */
@media (max-width: 1024px) {
    .getting-started-section {
        padding: 3rem 1.5rem;
    }
    
    .info-card {
        flex-direction: column;
        text-align: center;
    }
    
    .card-icon {
        align-self: center;
    }
}

@media (max-width: 768px) {
    .getting-started-hero {
        padding: 3rem 1.5rem;
        min-height: 250px;
    }
    
    .getting-started-hero h1 {
        font-size: 3.75rem;
        margin-bottom: 1rem;
    }
    
    .hero-description {
        font-size: 1.875rem;
        line-height: 1.6;
    }
    
    .getting-started-section {
        padding: 2rem 1rem;
    }
    
    .getting-started-section h2 {
        font-size: 2.5rem;
    }
    
    .setup-steps li {
        flex-direction: column;
        text-align: center;
        gap: 0.75rem;
    }
    
    .step-number {
        align-self: center;
    }
    
    .info-card {
        padding: 1.5rem;
    }
}

@media (max-width: 480px) {
    .getting-started-hero h1 {
        font-size: 3.125rem;
    }
    
    .hero-description {
        font-size: 1.5625rem;
    }
    
    .workflow-image img {
        max-width: 100%;
    }
}
</style> 