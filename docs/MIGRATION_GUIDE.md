# 🔄 Migration Guide: Manual to Automated API Documentation

## Overview

This guide helps you transition from manually maintained API documentation to the new automated system.

## What Changed?

### Before (Manual Process)
```
❌ Manually create .rst files for each module
❌ Manually update index.rst when adding modules  
❌ Easy to miss new modules
❌ Time-consuming maintenance
❌ Inconsistent formatting
```

### After (Automated Process)
```
✅ Automatically generate .rst files from code
✅ Automatically update index
✅ Never miss new modules
✅ One command to update all docs
✅ Consistent formatting
```

## Migration Steps

### 1. Backup Existing Documentation (Optional)

If you have custom content in existing API `.rst` files:

```bash
cd docs
mkdir -p _backup/api_old
cp -r api/* _backup/api_old/
```

### 2. Remove Old Manual Documentation

The automated system will replace these, so it's safe to remove:

```bash
cd docs
# Keep only hand-written content
# The generate_api_docs.sh script will handle the rest
```

### 3. Preserve Custom Content

If you have custom documentation you want to keep:

**Option A**: Move to separate directory
```bash
cd docs
mkdir -p custom_guides
mv api/your_custom_guide.rst custom_guides/
```

**Option B**: Add to main index manually
Edit `docs/index.rst` to include your custom pages.

### 4. Generate New Documentation

```bash
cd docs
./build_docs.sh --clean --open
```

### 5. Review Generated Documentation

Check that:
- ✅ All modules are documented
- ✅ Docstrings are rendering correctly
- ✅ Links work properly
- ✅ Custom content is preserved

## Handling Special Cases

### Case 1: Custom Module Documentation

If you had custom descriptions or examples in module `.rst` files:

**Solution**: Move them to the Python docstrings

**Before** (manual .rst):
```rst
topobench.mymodule
==================

This module does something special.

Example:
    >>> from topobench import mymodule
    >>> mymodule.do_something()
```

**After** (in Python code):
```python
"""My module - does something special.

This module provides functionality for...

Examples
--------
>>> from topobench import mymodule
>>> mymodule.do_something()
"""
```

### Case 2: Custom API Index Organization

If you had a custom-organized API index:

**Solution**: Modify `generate_api_index.py`

```python
def generate_api_index(api_dir, package_name):
    # ... existing code ...
    
    # Custom organization
    categories = {
        "Data": [m for m in modules if "data" in m],
        "Models": [m for m in modules if "model" in m],
        "Utilities": [m for m in modules if "utils" in m],
    }
    
    # Write organized index
    for category, mods in categories.items():
        f.write(f"\n{category}\n")
        f.write("-" * len(category) + "\n\n")
        for mod in sorted(mods):
            f.write(f"   {mod}\n")
```

### Case 3: Excluding Certain Modules

To exclude internal or experimental modules:

**Solution**: Edit `generate_api_docs.sh`

```bash
sphinx-apidoc \
    --force \
    --separate \
    -o "$API_DIR" \
    "$PROJECT_DIR/topobench" \
    "$PROJECT_DIR/topobench/__pycache__" \
    "$PROJECT_DIR/topobench/experimental"  # Add exclusions
    "$PROJECT_DIR/topobench/internal"
```

### Case 4: Custom Sphinx Directives

If you used custom directives in manual `.rst` files:

**Solution**: Keep those in separate files

```bash
docs/
├── api/                    # Auto-generated
├── custom/                 # Hand-written
│   └── advanced_usage.rst
└── index.rst              # Include both
```

## Best Practices After Migration

### 1. Documentation in Code
Put documentation in docstrings, not external files:

```python
class MyClass:
    """Short description.
    
    Longer description with details.
    
    Parameters
    ----------
    param : type
        Description
        
    Attributes
    ----------
    attr : type
        Description
        
    Examples
    --------
    >>> obj = MyClass(param=1)
    >>> obj.method()
    
    Notes
    -----
    Additional notes.
    
    See Also
    --------
    RelatedClass : Related functionality
    """
```

### 2. Regular Regeneration
Add to your workflow:

```bash
# Before committing Python changes
cd docs && make apidocs
git add docs/api/
git commit -m "Update API documentation"
```

### 3. CI/CD Integration
The system is already integrated with GitHub Actions - nothing to do! ✅

### 4. Review Generated Docs
Periodically review:
```bash
cd docs
./build_docs.sh --open
# Check for formatting issues, missing docstrings, etc.
```

## Comparison Table

| Aspect | Manual | Automated |
|--------|--------|-----------|
| New module | Edit .rst manually | Automatic |
| Update docs | Edit .rst files | Edit docstrings |
| Build time | Fast | Fast |
| Maintenance | High | Low |
| Consistency | Variable | High |
| Missing docs | Easy to miss | Impossible to miss |
| CI/CD | Manual setup | Built-in |

## Rollback Plan

If you need to rollback temporarily:

```bash
cd docs

# 1. Restore from backup
cp -r _backup/api_old/* api/

# 2. Modify Makefile to skip auto-generation
# Comment out the 'html: apidocs' line

# 3. Build normally
make html
```

## Getting Help

If you encounter issues:

1. **Check logs**: `/tmp/apidoc.log` and `/tmp/sphinx.log`
2. **Review docs**: Read `API_GENERATION.md`
3. **Test build**: Run `./build_docs.sh --clean --open`
4. **Verify imports**: Ensure packages are in `autodoc_mock_imports`

## Success Checklist

After migration, verify:

- ✅ `make html` works without errors
- ✅ All modules appear in documentation
- ✅ Docstrings render correctly
- ✅ Links between modules work
- ✅ GitHub Actions builds successfully
- ✅ Custom content is preserved
- ✅ Documentation deploys correctly

## Conclusion

The migration to automated API documentation:
- ⏱️ Saves time
- 🎯 Improves accuracy
- 📈 Scales effortlessly
- 🤖 Requires minimal maintenance

Welcome to automated documentation! 🎉
