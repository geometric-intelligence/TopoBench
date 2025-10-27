# 📚 TopoBench Documentation - Quick Reference

## 🚀 Quick Commands

### Build Documentation
```bash
cd docs
./build_docs.sh              # Build everything
./build_docs.sh --open       # Build and open in browser
./build_docs.sh --clean      # Clean and rebuild
./build_docs.sh --serve      # Build and start local server
```

### Manual Steps
```bash
cd docs
make apidocs                 # Generate API docs only
make html                    # Generate API + build HTML
make clean-api               # Clean API docs only
make clean                   # Clean all builds
```

### Development Mode
```bash
cd docs
./watch_and_regenerate.sh    # Auto-regenerate on file changes
```

## 📁 File Structure

```
docs/
├── build_docs.sh                    # 🆕 One-command builder
├── generate_api_docs.sh             # 🆕 API doc generator
├── watch_and_regenerate.sh          # 🆕 Development watcher
├── generate_api_index.py            # ✏️ Enhanced index generator
├── Makefile                         # ✏️ Updated with new targets
├── conf.py                          # Sphinx configuration
├── API_GENERATION.md               # 🆕 Complete guide
├── api/                            # ⚡ Auto-generated API docs
│   ├── index.rst                   # Auto-generated index
│   └── topobench.*.rst             # Auto-generated modules
└── _build/                         # Built documentation
```

## 🔧 Integration Points

### GitHub Actions (`.github/workflows/docs.yml`)
✅ Automatically generates and deploys docs on push to main

### Makefile Targets
- `make apidocs` - Generate API documentation
- `make clean-api` - Remove API documentation
- `make html` - Generate API + build HTML (auto-runs apidocs)
- `make clean` - Clean all builds

### Git Hooks
- `.git/hooks/pre-commit-api-docs-reminder` - Reminds to update docs

## 🎯 Common Tasks

### Task: Add New Module Documentation
```bash
# 1. Write your module with docstrings
# 2. Regenerate docs
cd docs && make apidocs
```

### Task: Review Documentation Locally
```bash
cd docs
./build_docs.sh --open
```

### Task: Debug Documentation Issues
```bash
cd docs
make clean
make clean-api
./build_docs.sh --clean --open
```

### Task: Continuous Development
```bash
# Terminal 1: Watch for changes
cd docs && ./watch_and_regenerate.sh

# Terminal 2: Serve docs
cd docs/_build/html && python -m http.server 8000
```

## 📝 Writing Good Docstrings

TopoBench uses NumPy-style docstrings:

```python
def my_function(param1, param2):
    """Short description of function.
    
    Longer description with more details about what
    the function does and why.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, optional
        Description of param2 (default is 42)
        
    Returns
    -------
    result : bool
        Description of return value
        
    Examples
    --------
    >>> my_function("hello", 5)
    True
    
    Notes
    -----
    Additional notes about edge cases or behavior.
    """
    pass
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors during build | Add package to `autodoc_mock_imports` in `conf.py` |
| Missing modules | Check module has `__init__.py` |
| Outdated docs | Run `make clean-api && make apidocs` |
| Build errors | Check `/tmp/sphinx.log` |

## 📚 Resources

- [Full Documentation Guide](API_GENERATION.md)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

## 🎉 Tips

- **Before Committing**: Run `make apidocs` if you changed Python files
- **CI/CD**: Docs auto-deploy on push to main
- **Development**: Use `./watch_and_regenerate.sh` for live updates
- **Testing**: Use `./build_docs.sh --serve` for local preview

---

**Generated**: October 27, 2025
**Automation Level**: 🤖 Fully Automated
