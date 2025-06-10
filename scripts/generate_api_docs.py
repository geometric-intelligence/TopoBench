#!/usr/bin/env python3
import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional

class APIDocGenerator:
    def __init__(self, library_path: str, output_dir: str):
        """Initialize the API documentation generator.
        
        Args:
            library_path: Path to the TopoBench library
            output_dir: Directory where documentation will be generated
        """
        self.library_path = Path(library_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_module(self, module_path: Path) -> Dict:
        """Parse a Python module and extract its documentation.
        
        Args:
            module_path: Path to the Python module
            
        Returns:
            Dictionary containing module documentation
        """
        module_name = module_path.stem
        with open(module_path, 'r') as f:
            tree = ast.parse(f.read())
        
        module_doc = {
            'name': module_name,
            'docstring': ast.get_docstring(tree) or '',
            'classes': [],
            'functions': []
        }
        
        # Parse classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or '',
                    'methods': [],
                    'attributes': []
                }
                
                # Parse methods
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        method_doc = {
                            'name': method.name,
                            'docstring': ast.get_docstring(method) or '',
                            'signature': self._get_signature(method)
                        }
                        class_doc['methods'].append(method_doc)
                
                # Parse class attributes
                for attr in node.body:
                    if isinstance(attr, ast.Assign):
                        for target in attr.targets:
                            if isinstance(target, ast.Name):
                                class_doc['attributes'].append({
                                    'name': target.id,
                                    'type': self._get_type(attr.value)
                                })
                
                module_doc['classes'].append(class_doc)
            
            # Parse functions
            elif isinstance(node, ast.FunctionDef):
                function_doc = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or '',
                    'signature': self._get_signature(node)
                }
                module_doc['functions'].append(function_doc)
        
        return module_doc
    
    def _get_signature(self, node: ast.FunctionDef) -> str:
        """Get the signature of a function or method.
        
        Args:
            node: AST node of the function or method
            
        Returns:
            String representation of the signature
        """
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return f"({', '.join(args)})"
    
    def _get_type(self, node: ast.AST) -> str:
        """Get the type of an AST node.
        
        Args:
            node: AST node
            
        Returns:
            String representation of the type
        """
        if isinstance(node, ast.Num):
            return type(node.n).__name__
        elif isinstance(node, ast.Str):
            return 'str'
        elif isinstance(node, ast.List):
            return 'list'
        elif isinstance(node, ast.Dict):
            return 'dict'
        elif isinstance(node, ast.Tuple):
            return 'tuple'
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return 'Any'
    
    def generate_markdown(self, module_doc: Dict) -> str:
        """Generate markdown documentation for a module.
        
        Args:
            module_doc: Module documentation dictionary
            
        Returns:
            Markdown string
        """
        markdown = []
        
        # Module title and docstring
        markdown.append(f"# {module_doc['name']}\n")
        if module_doc['docstring']:
            markdown.append(f"{module_doc['docstring']}\n")
        
        # Classes
        if module_doc['classes']:
            markdown.append("## Classes\n")
            for class_doc in module_doc['classes']:
                markdown.append(f"### {class_doc['name']}\n")
                if class_doc['docstring']:
                    markdown.append(f"{class_doc['docstring']}\n")
                
                # Methods
                if class_doc['methods']:
                    markdown.append("#### Methods\n")
                    for method in class_doc['methods']:
                        markdown.append(f"##### {method['name']}{method['signature']}\n")
                        if method['docstring']:
                            markdown.append(f"{method['docstring']}\n")
                
                # Attributes
                if class_doc['attributes']:
                    markdown.append("#### Attributes\n")
                    for attr in class_doc['attributes']:
                        markdown.append(f"- **{attr['name']}** ({attr['type']})\n")
        
        # Functions
        if module_doc['functions']:
            markdown.append("## Functions\n")
            for func in module_doc['functions']:
                markdown.append(f"### {func['name']}{func['signature']}\n")
                if func['docstring']:
                    markdown.append(f"{func['docstring']}\n")
        
        return "\n".join(markdown)
    
    def generate_documentation(self):
        """Generate documentation for all Python modules in the library."""
        # Find all Python files
        python_files = list(self.library_path.rglob("*.py"))
        
        for py_file in python_files:
            # Skip test files and __init__.py
            if "test" in str(py_file) or py_file.name == "__init__.py":
                continue
                
            # Parse module
            module_doc = self.parse_module(py_file)
            if not module_doc['classes'] and not module_doc['functions']:
                continue
                
            # Generate markdown
            markdown = self.generate_markdown(module_doc)
            
            # Write to file
            output_path = self.output_dir / f"{module_doc['name']}.md"
            output_path.write_text(markdown)
            
            print(f"Generated documentation for {module_doc['name']}")

def main():
    # Paths
    library_path = "topobench"
    output_dir = "_docs/documentation/api"
    
    # Generate documentation
    generator = APIDocGenerator(library_path, output_dir)
    generator.generate_documentation()

if __name__ == "__main__":
    main() 