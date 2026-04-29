# Contributing

Thank you for contributing to **arrowspace-tuner**!

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/). Every commit message must follow this format:

```
<type>: <short summary>

[optional body]
```

### Types

| Type | When to use |
|---|---|
| `feat` | New feature or behaviour |
| `fix` | Bug fix |
| `test` | Adding or fixing tests |
| `refactor` | Code change with no behaviour change |
| `chore` | Tooling, CI, dependencies, repo hygiene |
| `docs` | Documentation only |
| `perf` | Performance improvement |

### Examples

```
feat: add early-stopping to EpsTuner.fit()
fix: catch BaseException around ArrowSpace .build() for Rust panics
test: add degenerate-corpus fixture for pruning paths
chore: update .gitignore, remove .coverage artefact
docs: add quickstart section to README
```

### Rules

- Summary line ≤ 72 characters
- Use the imperative mood: "add", not "added" or "adds"
- Body explains **why**, not what (the diff shows the what)
- Reference issues/PRs in the body: `Fixes #12`

## Branch Names

```
feat/<short-description>
fix/<short-description>
chore/<short-description>
```

## Pull Requests

- All PRs must pass CI (pytest + ruff + mypy) before merging
- Squash-merge into `main`
- PR title must follow the same conventional commit format
