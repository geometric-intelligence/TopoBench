# Contributor agreement implementation plan

**Goal:** Incorporate a royalty-free commercial contribution grant into the
TopoBench PR process, using Topological Intelligence as the public project name.

**Architecture:** Keep the research licence separate from a versioned contributor
agreement. The contributor must receive a statement identifying the actual grant
recipients before accepting. Private records hold the identities and acceptance;
a public check verifies a maintainer's confirmation for the current PR commit.

**Tech stack:** Markdown, reStructuredText, GitHub Actions, GitHub REST API and
Node's built-in test runner. No runtime package changes.

## Design decisions

- Preserve contributor ownership and grant the identified recipients commercial
  sublicensing rights without contributor royalties for authorised uses.
- Use the project name and contact email publicly. An informal team name or
  control of an inbox does not substitute for identifying the recipients.
- Require affirmative acceptance of the agreement and disclosed recipient
  statement. A PR alone does not constitute completed commercial clearance.
- Keep the recipient information and acceptance records outside the public
  repository; use an opaque reference in the verification comment.
- Treat verification as a maintainer's review of those records, not an automatic
  determination of copyright ownership or legal enforceability.
- Invalidate verification when the PR commit changes; support explicit revocation.
- Run the check from trusted base-branch code, without executing PR code or
  publishing contributor identities. Do not merge this PR or change protection
  settings as part of this implementation.

## Implementation

1. Add `CLA.md` and align `CONTRIBUTING.md`, the PR template and contribution docs.
2. Replace the personal licensor heading with the public project identity in
   `LICENSE`; retain the definition of each actual rights holder as a Licensor.
   Preserve all publication authors, copyright notices and third-party licences.
3. Add the verifier and workflow, with tests for missing acceptance, missing or
   untrusted verification, stale commits, revocation and successful verification.
4. Document the recipient-statement and acceptance-record process, the automated
   check's limits, and the requirement to enable merge protection after deployment.
5. Retain the setuptools build minimum needed by the new PEP 639 metadata and add
   its rationale to the configuration and PR description.
6. Run verifier tests, pre-commit, focused documentation validation and package
   builds; inspect the packaged licence files. Review and push to PR #429.

## Adoption requirements

Supply and disclose a real recipient statement before collecting effective CLA
acceptances. Retain the contributor's exact accepted terms and authority records.
After the workflow is deployed, require the `Contributor agreement` status check
in branch protection or an equivalent ruleset. The existing hosting exception
decision remains open and is not changed by this work.
