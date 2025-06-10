#!/usr/bin/env python3
import os
from pathlib import Path
import re

def add_front_matter(file_path: Path):
    """Add Jekyll front matter to the markdown file."""
    content = file_path.read_text()
    
    # Extract the module name from the file path
    module_name = file_path.stem
    
    # Create front matter
    front_matter = f"""---
title: {module_name}
permalink: /docs/documentation/api/{module_name}/
layout: docs
---

"""
    
    # Add front matter to the content
    file_path.write_text(front_matter + content)

def add_styling(file_path: Path):
    """Add styling to the markdown file."""
    content = file_path.read_text()
    
    # Add styling section
    styling = """
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

h2 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
}

h3 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 1.5rem 0 1rem;
}

h4 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 1.25rem 0 0.75rem;
}

h5 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 1rem 0 0.5rem;
}

p {
    color: #4a5568;
    line-height: 1.6;
    margin-bottom: 1rem;
}

ul, ol {
    color: #4a5568;
    margin-bottom: 1rem;
    padding-left: 1.5rem;
}

li {
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

.method-signature {
    font-family: 'Fira Code', monospace;
    background: #f8fafc;
    padding: 0.5rem;
    border-radius: 4px;
    margin: 0.5rem 0;
}

.attribute-list {
    margin: 1rem 0;
}

.attribute-item {
    display: flex;
    align-items: baseline;
    margin-bottom: 0.5rem;
}

.attribute-name {
    font-weight: 600;
    color: #1a1a1a;
    margin-right: 0.5rem;
}

.attribute-type {
    color: #64748b;
    font-size: 0.875rem;
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

    h2 {
        font-size: 1.5rem;
    }

    h3 {
        font-size: 1.25rem;
    }
}
</style>
"""
    
    # Add styling to the content
    file_path.write_text(content + styling)

def format_code_blocks(content: str) -> str:
    """Format code blocks in the content."""
    # Find code blocks
    code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
    
    for block in code_blocks:
        # Replace with styled code block
        formatted_block = f'<div class="code-block">\n<pre><code>{block.strip()}</code></pre>\n</div>'
        content = content.replace(f'```python\n{block}\n```', formatted_block)
    
    return content

def format_method_signatures(content: str) -> str:
    """Format method signatures in the content."""
    # Find method signatures
    signatures = re.findall(r'##### (.*?)\n', content)
    
    for sig in signatures:
        # Replace with styled signature
        formatted_sig = f'<div class="method-signature">{sig}</div>'
        content = content.replace(f'##### {sig}\n', f'##### {sig}\n{formatted_sig}\n')
    
    return content

def format_attributes(content: str) -> str:
    """Format attributes in the content."""
    # Find attribute lists
    attribute_sections = re.findall(r'#### Attributes\n(.*?)(?=\n####|\n##|\n#|$)', content, re.DOTALL)
    
    for section in attribute_sections:
        # Find individual attributes
        attributes = re.findall(r'- \*\*(.*?)\*\* \((.*?)\)', section)
        
        formatted_attributes = '<div class="attribute-list">\n'
        for name, type_ in attributes:
            formatted_attributes += f'<div class="attribute-item">\n'
            formatted_attributes += f'<span class="attribute-name">{name}</span>\n'
            formatted_attributes += f'<span class="attribute-type">{type_}</span>\n'
            formatted_attributes += '</div>\n'
        formatted_attributes += '</div>'
        
        content = content.replace(section, formatted_attributes)
    
    return content

def main():
    # Path to the API documentation directory
    api_dir = Path("_docs/documentation/api")
    
    # Process each markdown file
    for file_path in api_dir.glob("*.md"):
        # Add front matter
        add_front_matter(file_path)
        
        # Read and format content
        content = file_path.read_text()
        
        # Format code blocks
        content = format_code_blocks(content)
        
        # Format method signatures
        content = format_method_signatures(content)
        
        # Format attributes
        content = format_attributes(content)
        
        # Write formatted content
        file_path.write_text(content)
        
        # Add styling
        add_styling(file_path)
        
        print(f"Formatted documentation for {file_path.name}")

if __name__ == "__main__":
    main() 