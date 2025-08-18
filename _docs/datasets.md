---
title: Datasets
permalink: /docs/datasets/
layout: docs
---

<div class="page-container">
    <!-- Filtering Sidebar -->
    <div class="filters-sidebar">
        <div class="search-container">
            <div class="search-wrapper">
                <i class="fas fa-search search-icon"></i>
                <input type="text" id="dataset-search" placeholder="Search for datasets" class="search-input">
            </div>
        </div>

        <div class="filter-section">
            <h3>Filter by Type</h3>
            <div class="filter-group" id="type-filters">
                <label class="filter-option">
                    <input type="checkbox" value="graph">
                    <span class="checkbox-custom"></span>
                    <span class="filter-label">Graph</span>
                    <span class="count">17</span>
                </label>
                <label class="filter-option">
                    <input type="checkbox" value="hypergraph">
                    <span class="checkbox-custom"></span>
                    <span class="filter-label">HyperGraph</span>
                    <span class="count">10</span>
                </label>
                <label class="filter-option">
                    <input type="checkbox" value="pointcloud">
                    <span class="checkbox-custom"></span>
                    <span class="filter-label">PointCloud</span>
                    <span class="count">1</span>
                </label>
                <label class="filter-option">
                    <input type="checkbox" value="simplicial">
                    <span class="checkbox-custom"></span>
                    <span class="filter-label">Simplicial</span>
                    <span class="count">4</span>
                </label>
            </div>
        </div>

        <div class="filter-section">
            <h3>Filter by Task Level</h3>
            <div class="filter-group" id="task-level-filters">
                <label class="filter-option">
                    <input type="checkbox" value="graph">
                    <span class="checkbox-custom"></span>
                    <span class="filter-label">Graph</span>
                    <span class="count">11</span>
                </label>
                <label class="filter-option">
                    <input type="checkbox" value="node">
                    <span class="checkbox-custom"></span>
                    <span class="filter-label">Node</span>
                    <span class="count">20</span>
                </label>
                <label class="filter-option">
                    <input type="checkbox" value="point-cloud">
                    <span class="checkbox-custom"></span>
                    <span class="filter-label">Point Cloud</span>
                    <span class="count">1</span>
                </label>
            </div>
        </div>

        <div class="active-filters" id="active-filters">
            <h3 class="active-filters-title" style="display: none;">Active Filters</h3>
            <div class="active-filters-list"></div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="datasets-container">
        <h1>Datasets</h1>

        <div class="intro-text">
            <p>TopoBench provides a comprehensive collection of datasets across different topological domains. Below you'll find our curated collection of datasets that can be used for benchmarking and research.</p>
        </div>

        <div class="datasets-grid">
            <div class="domain-section">
                <h2>Graph Datasets</h2>
                <div class="datasets-list">
                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/amazon-ratings" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Amazon Ratings</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/aosol" class="dataset-link">
                            <div class="dataset-header">
                                <h3>AOSOL</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/citeseer" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Citeseer</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/cora" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Cora</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/pubmed" class="dataset-link">
                            <div class="dataset-header">
                                <h3>PubMed</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/imdb-binary" class="dataset-link">
                            <div class="dataset-header">
                                <h3>IMDB Binary</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/imdb-multi" class="dataset-link">
                            <div class="dataset-header">
                                <h3>IMDB Multi</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/minesweeper" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Minesweeper</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/nci1" class="dataset-link">
                            <div class="dataset-header">
                                <h3>NCI1</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/nci109" class="dataset-link">
                            <div class="dataset-header">
                                <h3>NCI109</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/proteins" class="dataset-link">
                            <div class="dataset-header">
                                <h3>PROTEINS</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/questions" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Questions</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/reddit-binary" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Reddit Binary</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/roman_empire" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Roman Empire</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/tolokers" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Tolokers</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/us-county-demos" class="dataset-link">
                            <div class="dataset-header">
                                <h3>US County Demos</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/zinc" class="dataset-link">
                            <div class="dataset-header">
                                <h3>ZINC</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Graph</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>
                </div>
            </div>

            <div class="domain-section">
                <h2>HyperGraph Datasets</h2>
                <div class="datasets-list">
                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/cocitation-citeseer" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Cocitation Citeseer</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/coauthor-cora" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Coauthor Cora</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/cocitation-cora" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Cocitation Cora</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/cocitation-pubmed" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Cocitation PubMed</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/zoo" class="dataset-link">
                            <div class="dataset-header">
                                <h3>ZOO</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/ntu2012" class="dataset-link">
                            <div class="dataset-header">
                                <h3>NTU2012</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/dblp-ca" class="dataset-link">
                            <div class="dataset-header">
                                <h3>DBLP-CA</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/modelnet40" class="dataset-link">
                            <div class="dataset-header">
                                <h3>ModelNet40</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/mushroom" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Mushroom</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>

                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/20newsw100" class="dataset-link">
                            <div class="dataset-header">
                                <h3>20newsW100</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">HyperGraph</span>
                                <span class="task-tag">Task Level: Node</span>
                            </div>
                        </a>
                    </div>
                </div>
            </div>

            <div class="domain-section">
                <h2>PointCloud Datasets</h2>
                <div class="datasets-list">
                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/geometric-shapes" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Geometric Shapes</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">PointCloud</span>
                                <span class="task-tag">Task Level: Point Cloud</span>
                            </div>
                        </a>
                    </div>
                </div>
            </div>

            <div class="domain-section">
                <h2>Simplicial Datasets</h2>
                <div class="datasets-list">
                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/simplicial/mantra-betti-numbers" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Mantra Betti Numbers</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Simplicial</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>
                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/simplicial/mantra-genus" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Mantra Genus</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Simplicial</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>
                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/simplicial/mantra-name" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Mantra Name</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Simplicial</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>
                    <div class="dataset-card">
                        <a href="{{ site.baseurl }}/docs/datasets/simplicial/mantra-orientation" class="dataset-link">
                            <div class="dataset-header">
                                <h3>Mantra Orientation</h3>
                            </div>
                            <div class="dataset-tags">
                                <span class="domain-tag">Simplicial</span>
                                <span class="task-tag">Task Level: Graph</span>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
/* Page Layout */
.page-container {
    display: flex;
    gap: 2.5rem;
    max-width: 1600px;
    margin: 0 auto;
    padding: 2.5rem;
    background: #f8fafc;
}

/* Filtering Sidebar */
.filters-sidebar {
    width: 300px;
    flex-shrink: 0;
    background: #ffffff;
    border-radius: 12px;
    padding: 1.75rem;
    height: fit-content;
    position: sticky;
    top: 2rem;
    border: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 
                0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

/* Search Container */
.search-container {
    margin-bottom: 2rem;
}

.search-wrapper {
    position: relative;
    margin-bottom: 1rem;
}

.search-icon {
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    color: #64748b;
}

.search-input {
    width: 100%;
    padding: 0.875rem 1rem 0.875rem 2.5rem;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 8px;
    font-size: 0.9375rem;
    background: #ffffff;
    transition: all 0.2s ease;
    box-shadow: none;
    -webkit-appearance: none;
    appearance: none;
}

.search-input::-webkit-search-decoration,
.search-input::-webkit-search-cancel-button,
.search-input::-webkit-search-results-button,
.search-input::-webkit-search-results-decoration {
    display: none;
}

.search-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
    background: #ffffff;
}

/* Filter Sections */
.filter-section {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}

.filter-section h3 {
    font-size: 0.8125rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.filter-group {
    display: flex;
    flex-direction: column;
    gap: 0.875rem;
}

.filter-option {
    position: relative;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    cursor: pointer;
    padding: 0.375rem 0;
    transition: all 0.2s ease;
}

.filter-option:hover {
    color: #2563eb;
}

.filter-option input[type="checkbox"] {
    position: absolute;
    opacity: 0;
    cursor: pointer;
    height: 0;
    width: 0;
}

.checkbox-custom {
    position: relative;
    height: 1.125rem;
    width: 1.125rem;
    background-color: #ffffff;
    border: 2px solid #cbd5e1;
    border-radius: 4px;
    transition: all 0.2s ease;
}

.filter-option:hover .checkbox-custom {
    border-color: #2563eb;
}

.filter-option input:checked ~ .checkbox-custom {
    background-color: #2563eb;
    border-color: #2563eb;
}

.checkbox-custom:after {
    content: '';
    position: absolute;
    display: none;
    left: 5px;
    top: 2px;
    width: 4px;
    height: 8px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}

.filter-option input:checked ~ .checkbox-custom:after {
    display: block;
}

.filter-label {
    flex: 1;
    font-size: 0.9375rem;
    color: #475569;
    font-weight: 500;
}

.count {
    font-size: 0.75rem;
    color: #64748b;
    background: #f1f5f9;
    padding: 0.25rem 0.625rem;
    border-radius: 999px;
    font-weight: 500;
}

/* Active Filters */
.active-filters {
    margin-top: 1.5rem;
}

.active-filters-title {
    font-size: 0.8125rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.active-filters-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.active-filter {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.375rem 0.75rem;
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 999px;
    font-size: 0.8125rem;
    font-weight: 500;
}

.remove-filter {
    cursor: pointer;
    opacity: 0.7;
    transition: opacity 0.2s ease;
}

.remove-filter:hover {
    opacity: 1;
}

/* Main Content Container */
.datasets-container {
    flex: 1;
    background: #ffffff;
    border-radius: 12px;
    padding: 2rem;
    border: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Responsive Design */
@media (max-width: 1200px) {
    .page-container {
        padding: 1.5rem;
        gap: 1.5rem;
    }
    
    .filters-sidebar {
        width: 260px;
    }
}

@media (max-width: 1024px) {
    .page-container {
        flex-direction: column;
    }

    .filters-sidebar {
        width: 100%;
        position: static;
        margin-bottom: 1.5rem;
    }
}

/* Container and Global Styles */
.datasets-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
    background-color: #ffffff;
}

/* Typography */
h1 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #1a1a1a;
    margin-bottom: 2rem;
    letter-spacing: -0.03em;
    text-align: center;
}

.intro-text {
    max-width: 800px;
    margin: 0 auto 3rem;
    text-align: center;
}

.intro-text p {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 1.125rem;
    line-height: 1.6;
    color: #4a5568;
    margin-bottom: 1rem;
    font-weight: 400;
}

/* Domain Section */
.domain-section {
    margin-bottom: 3rem;
}

.domain-section h2 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 1.5rem;
    letter-spacing: -0.02em;
}

/* Datasets Grid */
.datasets-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1rem;
}

/* Dataset Cards */
.dataset-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04), 
                0 4px 8px rgba(0, 0, 0, 0.02);
    transition: all 0.2s cubic-bezier(0.165, 0.84, 0.44, 1);
    border: 1px solid rgba(0, 0, 0, 0.06);
    cursor: pointer;
}

.dataset-link {
    text-decoration: none;
    color: inherit;
    display: block;
}

.dataset-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08), 
                0 8px 16px rgba(0, 0, 0, 0.04);
}

.dataset-header {
    margin-bottom: 0.75rem;
}

.dataset-card h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
    letter-spacing: -0.01em;
}

/* Tags */
.dataset-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.domain-tag, .task-tag {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.025em;
}

.domain-tag {
    background: #ebf5ff;
    color: #3b82f6;
}

.task-tag {
    background: #f0fdf4;
    color: #16a34a;
}

/* Responsive Design */
@media (max-width: 1200px) {
    .datasets-container {
        padding: 1.5rem;
    }

    .datasets-list {
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    }
}

@media (max-width: 768px) {
    h1 {
        font-size: 2.5rem;
    }

    .domain-section h2 {
        font-size: 1.5rem;
    }

    .datasets-list {
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }

    .dataset-card {
        padding: 1rem;
    }
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('dataset-search');
    const typeFilters = document.getElementById('type-filters').querySelectorAll('input');
    const taskLevelFilters = document.getElementById('task-level-filters').querySelectorAll('input');
    const cards = document.querySelectorAll('.dataset-card');
    const activeFiltersList = document.querySelector('.active-filters-list');
    const activeFiltersTitle = document.querySelector('.active-filters-title');
    const domainSections = document.querySelectorAll('.domain-section');

    function updateActiveFilters() {
        activeFiltersList.innerHTML = '';
        
        const activeFilters = [];
        
        // Add type filters
        Array.from(typeFilters)
            .filter(input => input.checked)
            .forEach(input => {
                activeFilters.push(addActiveFilter(input.value, 'type'));
            });

        // Add task level filters
        Array.from(taskLevelFilters)
            .filter(input => input.checked)
            .forEach(input => {
                activeFilters.push(addActiveFilter(input.value, 'task'));
            });

        // Show/hide the "Active Filters" title based on whether there are any active filters
        activeFiltersTitle.style.display = activeFilters.length > 0 ? 'block' : 'none';
    }

    function addActiveFilter(value, type) {
        const filter = document.createElement('span');
        filter.className = 'active-filter';
        const displayValue = value.replace('-', ' ').replace(/\b\w/g, c => c.toUpperCase());
        filter.innerHTML = `
            ${displayValue}
            <span class="remove-filter" data-value="${value}" data-type="${type}">×</span>
        `;
        activeFiltersList.appendChild(filter);

        filter.querySelector('.remove-filter').addEventListener('click', function() {
            const filterValue = this.dataset.value;
            const filterType = this.dataset.type;
            
            const checkbox = filterType === 'type' 
                ? Array.from(typeFilters).find(input => input.value === filterValue)
                : Array.from(taskLevelFilters).find(input => input.value === filterValue);
                
            if (checkbox) {
                checkbox.checked = false;
                filterDatasets();
            }
        });

        return filter;
    }

    function getTaskLevelFromTag(taskTag) {
        const match = taskTag.match(/Task Level: (.*)/i);
        if (!match) return '';
        
        // Normalize the task level text to match the filter values
        const taskLevel = match[1].toLowerCase();
        if (taskLevel === 'point cloud') {
            return 'point-cloud';
        }
        return taskLevel;
    }

    function filterDatasets() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedTypes = Array.from(typeFilters)
            .filter(input => input.checked)
            .map(input => input.value);
        const selectedTaskLevels = Array.from(taskLevelFilters)
            .filter(input => input.checked)
            .map(input => input.value);

        updateActiveFilters();

        // Track visible cards in each section
        const visibleCardsInSection = new Map();

        // First pass: determine which cards should be visible and count them per section
        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            const type = card.querySelector('.domain-tag').textContent.toLowerCase();
            const taskTag = card.querySelector('.task-tag').textContent;
            const taskLevel = getTaskLevelFromTag(taskTag);
            
            const matchesSearch = text.includes(searchTerm);
            const matchesType = selectedTypes.length === 0 || selectedTypes.some(t => type.includes(t));
            const matchesTaskLevel = selectedTaskLevels.length === 0 || 
                selectedTaskLevels.includes(taskLevel);

            const isVisible = matchesSearch && matchesType && matchesTaskLevel;
            
            // Get the parent section
            const section = card.closest('.domain-section');
            if (section) {
                const sectionId = section.querySelector('h2').textContent;
                visibleCardsInSection.set(
                    sectionId, 
                    (visibleCardsInSection.get(sectionId) || 0) + (isVisible ? 1 : 0)
                );
            }

            card.style.display = isVisible ? 'block' : 'none';
        });

        // Second pass: show/hide sections based on whether they have visible cards
        domainSections.forEach(section => {
            const sectionId = section.querySelector('h2').textContent;
            const hasVisibleCards = visibleCardsInSection.get(sectionId) > 0;
            section.style.display = hasVisibleCards ? 'block' : 'none';
            
            // Only show section title if no type filters are active or if this section matches the type filter
            const sectionTitle = section.querySelector('h2');
            if (sectionTitle) {
                const isFiltering = selectedTypes.length > 0 || selectedTaskLevels.length > 0 || searchTerm.length > 0;
                sectionTitle.style.display = isFiltering ? 'none' : 'block';
            }
        });

        // Update grid layout after filtering
        const datasetsLists = document.querySelectorAll('.datasets-list');
        datasetsLists.forEach(list => {
            list.style.gridTemplateColumns = 'repeat(auto-fill, minmax(250px, 1fr))';
        });
    }

    // Event listeners
    searchInput.addEventListener('input', filterDatasets);
    typeFilters.forEach(filter => filter.addEventListener('change', filterDatasets));
    taskLevelFilters.forEach(filter => filter.addEventListener('change', filterDatasets));

    // Initial filter update
    filterDatasets();
});
</script> 