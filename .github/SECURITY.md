# Security Policy

## Supported Versions

Security fixes go into the latest release only.

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in talkpipe-agents, please report
it responsibly:

1. **Do not** open a public GitHub issue for it.
2. Email the report to tlbauer@sandia.gov.
3. Include a description of the vulnerability, steps to reproduce it, the
   potential impact, and a suggested fix if you have one.

A vulnerability in the agent's tool suite (for example a tool that reads
files or fetches URLs doing more than its documentation says) counts; one in
[talkpipe](https://github.com/sandialabs/talkpipe) or in
[Strands Agents](https://github.com/strands-agents/sdk-python) belongs to
that project.

### What to Expect

- **Response**: we aim to acknowledge receipt within 2 business days.
- **Assessment**: an initial assessment within 5 business days.
- **Updates**: progress updates every 7 days until resolution.
- **Disclosure**: public disclosure is coordinated with the reporter once a
  fix is released.

### Security Measures

- Dependency scanning with Safety and static analysis with Bandit on every
  CI run, both gating the build.
- Static code analysis with CodeQL.
- Automated dependency updates via Dependabot.
- Declared dependency floors are security minimums: the lowest release of
  each dependency with no known advisory.

### Acknowledgments

Researchers who responsibly disclose vulnerabilities are acknowledged in the
release notes unless they prefer to remain anonymous.
