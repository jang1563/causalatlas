# Security Policy

## Supported Versions

The current `main` branch and the latest GitHub release are supported for
software, metadata, and public-release-surface issues.

## Reporting

For software vulnerabilities, accidental exposure of secrets, or release-surface
problems that should not be posted publicly, use GitHub's security advisory
workflow for this repository when available. For ordinary reproducibility bugs,
documentation corrections, or artifact-schema issues, open a public GitHub
issue.

Please do not include access tokens, unpublished datasets, identifiable human
subject data, or large upstream data files in an issue or report. Instead,
describe the affected path, command, artifact, and expected behavior.

## Release-Surface Issues

This repository intentionally excludes raw third-party data, local run logs,
large regenerated intermediates, and service credentials. Reports that identify
accidental inclusion of those materials are treated as release-blocking issues.
