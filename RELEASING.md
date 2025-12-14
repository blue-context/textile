# Releasing Textile to PyPI

This document describes the process for releasing new versions of Textile to PyPI.

## Prerequisites

1. **PyPI Trusted Publishing** must be configured:
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher for `blue-context/textile`
   - Workflow name: `publish.yml`
   - Environment: leave blank or use `release`

2. **Permissions**: Must have maintainer access to the repository

## Release Process

### 1. Update Version

Edit `pyproject.toml` and update the version:

```toml
[project]
name = "textile"
version = "0.4.0"  # Update this
```

### 2. Update Changelog (Optional)

If maintaining a CHANGELOG.md, add release notes for the new version.

### 3. Commit and Push

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.4.0"
git push origin main
```

### 4. Create and Push Tag

```bash
# Create annotated tag
git tag -a v0.4.0 -m "Release v0.4.0"

# Push tag to GitHub
git push origin v0.4.0
```

### 5. Create GitHub Release

1. Go to https://github.com/blue-context/textile/releases/new
2. Select the tag you just pushed (e.g., `v0.4.0`)
3. Title: `v0.4.0`
4. Description: Add release notes describing changes
5. Click "Publish release"

### 6. Automated Publishing

Once the GitHub Release is published:
- The `.github/workflows/publish.yml` workflow will trigger automatically
- It will build the package using `uv build`
- It will publish to PyPI using trusted publishing (no API token needed)
- Monitor the workflow at: https://github.com/blue-context/textile/actions

### 7. Verify Publication

Check that the new version appears on PyPI:
- https://pypi.org/project/textile/

Test installation:

```bash
pip install textile==0.4.0
```

## Troubleshooting

### Trusted Publishing Not Configured

If you see an error like "Trusted publishing exchange failure", you need to configure PyPI trusted publishing:

1. Go to https://pypi.org/manage/account/publishing/
2. Add publisher with these settings:
   - PyPI Project Name: `textile`
   - Owner: `blue-context`
   - Repository name: `textile`
   - Workflow name: `publish.yml`
   - Environment name: (leave blank)

### Build Fails

If the build fails, test locally first:

```bash
uv build
```

This will create files in `dist/` directory. Check for any errors.

### Version Already Exists on PyPI

PyPI doesn't allow re-uploading the same version. You must increment the version number:

- Patch: `0.3.0` → `0.3.1` (bug fixes)
- Minor: `0.3.0` → `0.4.0` (new features, backward compatible)
- Major: `0.3.0` → `1.0.0` (breaking changes)

## Version Scheme

Textile follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples:
- `0.3.0` → `0.3.1`: Bug fix release
- `0.3.0` → `0.4.0`: New transformer added
- `0.3.0` → `1.0.0`: API redesign (breaking changes)
