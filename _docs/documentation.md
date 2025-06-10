---
title: Documentation
permalink: /docs/documentation/
layout: docs
---

<div class="page-container">
    <div class="documentation-content">
        <h1>TopoBench Documentation</h1>

        <div class="intro-text">
            <p>TopoBench is a Python library designed to standardize benchmarking and accelerate research in Topological Deep Learning. This documentation provides comprehensive information about the library's features, usage, and implementation details.</p>
        </div>

        <div class="documentation-sections">
            <div class="section">
                <h2>Core Concepts</h2>
                <p>Learn about the fundamental concepts and architecture of TopoBench.</p>
                <ul>
                    <li><a href="{{ site.baseurl }}/docs/documentation/core-concepts">Topological Deep Learning</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/architecture">Library Architecture</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/data-structures">Data Structures</a></li>
                </ul>
            </div>

            <div class="section">
                <h2>Installation & Setup</h2>
                <p>Get started with TopoBench installation and configuration.</p>
                <ul>
                    <li><a href="{{ site.baseurl }}/docs/documentation/installation">Installation Guide</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/configuration">Configuration</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/environment">Environment Setup</a></li>
                </ul>
            </div>

            <div class="section">
                <h2>API Reference</h2>
                <p>Detailed documentation of TopoBench's core components.</p>
                <ul>
                    <li><a href="{{ site.baseurl }}/docs/documentation/api/datasets">Datasets API</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/api/models">Models API</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/api/transforms">Transforms API</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/api/evaluation">Evaluation API</a></li>
                </ul>
            </div>

            <div class="section">
                <h2>Usage Guides</h2>
                <p>Practical guides for using TopoBench in your projects.</p>
                <ul>
                    <li><a href="{{ site.baseurl }}/docs/documentation/guides/quickstart">Quick Start Guide</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/guides/datasets">Working with Datasets</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/guides/models">Building Models</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/guides/evaluation">Model Evaluation</a></li>
                </ul>
            </div>

            <div class="section">
                <h2>Advanced Topics</h2>
                <p>In-depth information about advanced features and customization.</p>
                <ul>
                    <li><a href="{{ site.baseurl }}/docs/documentation/advanced/custom-datasets">Custom Datasets</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/advanced/custom-models">Custom Models</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/advanced/benchmarking">Benchmarking</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/advanced/performance">Performance Optimization</a></li>
                </ul>
            </div>

            <div class="section">
                <h2>Examples & Tutorials</h2>
                <p>Learn through practical examples and tutorials.</p>
                <ul>
                    <li><a href="{{ site.baseurl }}/docs/documentation/examples/basic">Basic Examples</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/examples/advanced">Advanced Examples</a></li>
                    <li><a href="{{ site.baseurl }}/docs/documentation/examples/custom">Custom Implementations</a></li>
                </ul>
            </div>
        </div>

        <div class="additional-info">
            <h2>Additional Resources</h2>
            <div class="resources-grid">
                <div class="resource-card">
                    <h3>GitHub Repository</h3>
                    <p>Access the full source code and contribute to the project.</p>
                    <a href="https://github.com/geometric-intelligence/TopoBench" class="btn btn-primary">View on GitHub</a>
                </div>
                <div class="resource-card">
                    <h3>Research Paper</h3>
                    <p>Read the original research paper about TopoBench.</p>
                    <a href="https://arxiv.org/abs/2406.06642" class="btn btn-primary">Read Paper</a>
                </div>
                <div class="resource-card">
                    <h3>Contact Us</h3>
                    <p>Get in touch with the development team.</p>
                    <a href="mailto:topological.intelligence@gmail.com" class="btn btn-primary">Email Us</a>
                </div>
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
    text-align: center;
}

.intro-text {
    max-width: 800px;
    margin: 0 auto 3rem;
    text-align: center;
}

.intro-text p {
    font-size: 1.125rem;
    line-height: 1.6;
    color: #4a5568;
}

.documentation-sections {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
}

.section {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1.5rem;
    transition: transform 0.2s ease;
}

.section:hover {
    transform: translateY(-2px);
}

.section h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 1rem;
}

.section p {
    color: #4a5568;
    margin-bottom: 1rem;
}

.section ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.section ul li {
    margin-bottom: 0.5rem;
}

.section ul li a {
    color: #3b82f6;
    text-decoration: none;
    transition: color 0.2s ease;
}

.section ul li a:hover {
    color: #2563eb;
    text-decoration: underline;
}

.additional-info {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #e2e8f0;
}

.additional-info h2 {
    text-align: center;
    margin-bottom: 2rem;
}

.resources-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
}

.resource-card {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease;
}

.resource-card:hover {
    transform: translateY(-2px);
}

.resource-card h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 1rem;
}

.resource-card p {
    color: #4a5568;
    margin-bottom: 1.5rem;
}

.btn {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    background-color: #3b82f6;
    color: white;
    text-decoration: none;
    border-radius: 6px;
    transition: background-color 0.2s ease;
}

.btn:hover {
    background-color: #2563eb;
    color: white;
    text-decoration: none;
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

    .documentation-sections,
    .resources-grid {
        grid-template-columns: 1fr;
    }
}
</style> 