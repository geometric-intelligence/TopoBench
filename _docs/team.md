---
title: Team
permalink: /docs/team/
---

<div class="team-container">
    <h1>Our Team</h1>

    <div class="intro-text">
        <p>Welcome to the TopoBench team page! Here you'll find the dedicated researchers, engineers, and maintainers who are actively shaping the future of topological deep learning through their contributions to TopoBench.</p>
        
        <p>Our team brings together expertise from various domains including geometric deep learning, topological data analysis, and machine learning, working collaboratively to advance the field and maintain the quality and innovation of our platform.</p>
    </div>

    <div class="row team-row">
        <div class="col-md-4 team-member">
            <img src="{{ site.baseurl }}/assets/img/placeholder-profile.png" alt="Lev Telyatnikov" class="img-circle">
            <h4>Lev Telyatnikov</h4>
            <div class="social-links">
                <a href="https://github.com/levtelyatnikov" aria-label="GitHub"><i class="fa fa-github"></i></a>
                <a href="https://www.linkedin.com/in/lev-telyatnikov/" aria-label="LinkedIn"><i class="fa fa-linkedin"></i></a>
            </div>
        </div>
        <div class="col-md-4 team-member">
            <img src="{{ site.baseurl }}/assets/img/placeholder-profile.png" alt="Guillermo Bernardez" class="img-circle">
            <h4>Guillermo Bernardez</h4>
            <div class="social-links">
                <a href="https://github.com/guillermo-bernardez" aria-label="GitHub"><i class="fa fa-github"></i></a>
                <a href="https://www.linkedin.com/in/guillermo-bernardez/" aria-label="LinkedIn"><i class="fa fa-linkedin"></i></a>
            </div>
        </div>
        <div class="col-md-4 team-member">
            <img src="{{ site.baseurl }}/assets/img/placeholder-profile.png" alt="Marco Montagna" class="img-circle">
            <h4>Marco Montagna</h4>
            <div class="social-links">
                <a href="https://github.com/marcomontagna" aria-label="GitHub"><i class="fa fa-github"></i></a>
                <a href="https://www.linkedin.com/in/marco-montagna/" aria-label="LinkedIn"><i class="fa fa-linkedin"></i></a>
            </div>
        </div>
    </div>

    <div class="row team-row">
        <div class="col-md-4 team-member">
            <img src="{{ site.baseurl }}/assets/img/placeholder-profile.png" alt="Nina Miolane" class="img-circle">
            <h4>Nina Miolane</h4>
            <div class="social-links">
                <a href="https://github.com/ninamiolane" aria-label="GitHub"><i class="fa fa-github"></i></a>
                <a href="https://www.linkedin.com/in/nina-miolane/" aria-label="LinkedIn"><i class="fa fa-linkedin"></i></a>
                <a href="https://scholar.google.com/citations?user=61CTrYoAAAAJ" aria-label="Google Scholar"><i class="fa fa-graduation-cap"></i></a>
            </div>
        </div>
        <div class="col-md-4 team-member">
            <img src="{{ site.baseurl }}/assets/img/placeholder-profile.png" alt="Simone Scardapane" class="img-circle">
            <h4>Simone Scardapane</h4>
            <div class="social-links">
                <a href="https://github.com/simone-scardapane" aria-label="GitHub"><i class="fa fa-github"></i></a>
                <a href="https://scholar.google.com/citations?user=9FJOj-IAAAAJ" aria-label="Google Scholar"><i class="fa fa-graduation-cap"></i></a>
            </div>
        </div>
        <div class="col-md-4 team-member">
            <img src="{{ site.baseurl }}/assets/img/placeholder-profile.png" alt="Theodore Papamarkou" class="img-circle">
            <h4>Theodore Papamarkou</h4>
            <div class="social-links">
                <a href="https://github.com/theodore-papamarkou" aria-label="GitHub"><i class="fa fa-github"></i></a>
                <a href="https://scholar.google.com/citations?user=KUkBCe0AAAAJ" aria-label="Google Scholar"><i class="fa fa-graduation-cap"></i></a>
            </div>
        </div>
    </div>
</div>

<style>
/* Container and Global Styles */
.team-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    background-color: #ffffff;
}

/* Typography */
h1 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #1a1a1a;
    margin-bottom: 2rem;
    letter-spacing: -0.03em;
    text-align: center;
}

.intro-text {
    max-width: 800px;
    margin: 0 auto 4rem;
    text-align: center;
}

.intro-text p {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 1.25rem;
    line-height: 1.7;
    color: #4a5568;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

/* Team Grid Layout */
.team-row {
    margin: 0 -1.5rem 4rem;
    display: flex;
    justify-content: center;
    gap: 2.5rem;
}

/* Team Member Cards */
.team-member {
    text-align: center;
    padding: 2.5rem;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02), 
                0 10px 15px rgba(0, 0, 0, 0.03);
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    border: 1px solid rgba(0, 0, 0, 0.04);
}

.team-member:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.05), 
                0 20px 48px rgba(0, 0, 0, 0.025);
}

/* Profile Images */
.team-member img {
    width: 200px;
    height: 200px;
    margin-bottom: 2rem;
    border: 4px solid #ffffff;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
    transition: all 0.4s ease;
    border-radius: 50%;
}

.team-member img:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
}

/* Name Typography */
.team-member h4 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 1.5rem;
    margin: 1rem 0;
    color: #1a1a1a;
    font-weight: 700;
    letter-spacing: -0.01em;
}

/* Social Links */
.social-links {
    margin-top: 1.5rem;
    display: flex;
    justify-content: center;
    gap: 0.75rem;
}

.social-links a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    color: #4a5568;
    font-size: 1.25rem;
    border-radius: 50%;
    transition: all 0.3s ease;
    background: #f8fafc;
    border: 1px solid rgba(0, 0, 0, 0.04);
}

.social-links a:hover {
    color: #3b82f6;
    background: #eff6ff;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.1);
}

/* Specific Icon Adjustments */
.fa-graduation-cap {
    font-size: 1.125rem;
}

/* Responsive Design */
@media (max-width: 1024px) {
    .team-container {
        padding: 1.5rem;
    }

    h1 {
        font-size: 3rem;
    }

    .intro-text p {
        font-size: 1.125rem;
    }

    .team-member {
        padding: 2rem;
    }
}

@media (max-width: 768px) {
    h1 {
        font-size: 2.5rem;
    }

    .team-row {
        flex-direction: column;
        gap: 2rem;
    }

    .team-member {
        margin: 0 1rem;
    }

    .team-member img {
        width: 180px;
        height: 180px;
    }

    .team-member h4 {
        font-size: 1.35rem;
    }
}
</style>

<div class="community-section">
    <h2>Join Our Community</h2>
    <p>TopoBench is an open-source project that grows stronger with each new contributor. Whether you're interested in developing new features, improving documentation, or sharing your expertise, there's a place for you in our community.</p>
    
    <div class="contribution-steps">
        <h3>How to Contribute</h3>
        <ol>
            <li>Fork the repository</li>
            <li>Create a feature branch</li>
            <li>Submit a pull request</li>
            <li>Join our discussions</li>
        </ol>
    </div>
</div>

<style>
.community-section {
    margin-top: 6rem;
    padding: 4rem;
    background: linear-gradient(to bottom right, #f8fafc, #f1f5f9);
    border-radius: 24px;
    text-align: center;
}

.community-section h2 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 1.5rem;
    letter-spacing: -0.02em;
}

.community-section p {
    font-size: 1.25rem;
    line-height: 1.7;
    color: #4a5568;
    max-width: 800px;
    margin: 0 auto 3rem;
}

.contribution-steps {
    max-width: 600px;
    margin: 0 auto;
    text-align: left;
}

.contribution-steps h3 {
    font-size: 1.5rem;
    color: #1a1a1a;
    margin-bottom: 1.5rem;
    font-weight: 600;
}

.contribution-steps ol {
    list-style-type: none;
    counter-reset: steps;
    padding: 0;
}

.contribution-steps li {
    position: relative;
    padding-left: 3rem;
    margin-bottom: 1.25rem;
    font-size: 1.125rem;
    color: #4a5568;
    line-height: 1.6;
}

.contribution-steps li::before {
    counter-increment: steps;
    content: counter(steps);
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 32px;
    height: 32px;
    background: #3b82f6;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.875rem;
}

@media (max-width: 768px) {
    .community-section {
        padding: 3rem 1.5rem;
        margin-top: 4rem;
    }

    .community-section h2 {
        font-size: 2rem;
    }

    .community-section p {
        font-size: 1.125rem;
    }
}
</style> 