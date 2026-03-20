# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 3.x     | :white_check_mark: |
| < 3.0   | :x:                |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please use one of the following methods:

1. **Preferred**: [GitHub Private Vulnerability Reporting](https://github.com/tacktechai/mieza.meanings/security/advisories/new)
2. **Alternative**: Email security@mieza.ai

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Target fix**: Within 30 days for confirmed vulnerabilities

We will credit reporters in the release notes unless they prefer to remain anonymous.

## Scope

This policy covers the mieza.meanings library and its direct code, including:

- Clojure source code
- OpenCL kernel code
- Model serialization/deserialization
- Memory-mapped file handling

For vulnerabilities in transitive dependencies (e.g., Hadoop, Arrow, Jackson),
please report them to the upstream project. If a transitive dependency
vulnerability affects mieza.meanings users, we will work to update the
dependency promptly.
