# Security Policy

## Supported Versions

FlipRate maintains security updates for the current minor release series:

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| 0.8.x   | :white_check_mark: |
| 0.7.x   | :white_check_mark: |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Reporting a Vulnerability

We take the security of FlipRate seriously. If you believe you have found a security vulnerability in FlipRate:

1. **Do not disclose the vulnerability publicly** in a GitHub issue, pull request, discussion forum, or social media.
2. **Send a detailed report via email** to the maintainer:
   - **Email**: `shiv.supane777@gmail.com`
   - **Subject**: `[SECURITY] Vulnerability report for FlipRate`
3. Include in your report:
   - Type of issue (e.g. buffer overflow, denial of service, remote code execution, PII leakage, unauthenticated bypass)
   - Clear step-by-step reproduction instructions or a minimal reproducible code snippet
   - Versions of FlipRate, Python, and operating system tested
   - Proposed remediations or patches, if any

### Response Timeline

- **Initial Response**: Within 48 hours acknowledging receipt of the report.
- **Triage & Status Update**: Within 5 business days detailing validation and assessment.
- **Remediation**: Once verified, a security advisory and patched release will be coordinated.

## Privacy & Security Best Practices in FlipRate

- **Data Privacy**: FlipRate test reports and explainability tools default to omitting raw original and variant texts (`opt_in_examples=False` in `explain_report_flips()`) to avoid exposing sensitive PII.
- **Input Boundaries**: When using FlipRate's HTTP/API endpoints (`fliprate.api`), always configure `FLIPRATE_API_KEY` and restrict `FLIPRATE_CORS` in production.
- **Dependency Isolation**: Core FlipRate has zero runtime dependencies. Optional extras (`api`, `hf`, `sklearn`) should be installed only when needed for specific integrations.
- **Offline / Air-Gapped Operation**: Core normalizers, perturbation engines, and rule sets operate entirely locally in-memory without initiating external network requests.
