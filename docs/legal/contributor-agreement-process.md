# Contributor agreement records and verification

The [TopoBench Contributor Agreement 1.0](../../CLA.md) is the agreement.
This page explains how to collect and verify acceptance; it does not itself
grant rights or identify the recipients.

## Recipient statement

Topological Intelligence is currently an informal project name. Before collecting
acceptances, supply each contributor with a versioned statement identifying the
actual persons or organisation receiving the commercial grant. Include enough
information to distinguish them, their legal capacity and a contact address.
An internal list, an email address or an opaque reference alone is insufficient.
Do not describe the project name as an incorporated entity unless that is true.

Give the contributor the complete statement before acceptance. Retain the exact
statement with the accepted agreement version. Changing its recipients requires
new acceptance for the expanded grant; adding a repository administrator does
not add a commercial grant recipient.

## Acceptance record

Keep the following together in an access-controlled record available to the parties:

- The full agreement text and version that the contributor accepted.
- The full recipient statement, its version and evidence it was supplied first.
- The accepting person's identity, represented rights holder and authority.
- The repository, PR and commits or other precise contribution scope covered.
- The affirmative acceptance, its date and the evidence authenticating it.
- Any other rights holders' permissions and identified third-party material.

A PDF signature is not the only possible acceptance method. An appropriately
recorded electronic acceptance may be used, but a checkbox without the recipient
information or a maintainer's own assertion is not a substitute for actual assent.
A PR with several rights holders needs coverage for each, directly or through
an authorised representative. Existing records may cover later changes only when
their accepted scope actually includes those changes.

Use an opaque record identifier in public, without names, emails or other private
details encoded in it. Keep the associated records for as long as the project
relies on the grant; an identifier without its underlying evidence is insufficient.

## Maintainer verification

Before verification, check the PR's current commit, contribution scope, actual
rights holders, disclosed recipient statement and recorded affirmative acceptance.
Confirm separately licensed material is identified and its conditions are met.
Do not verify an incomplete record. The CLA does not clear rights in other code.

After verification, a maintainer with repository write, maintain or admin
permission posts a PR comment containing only this line:

```text
CLA-VERIFIED: <full-40-character-head-commit> <opaque-record-id>
```

The record identifier uses letters, digits, dots, underscores or hyphens, starts
with a letter or digit and contains at most 80 characters. Replace the angle-bracket
placeholders. A PR can have one record that collects all required acceptances.

To revoke the public verification, post:

```text
CLA-REVOKED: <full-40-character-head-commit>
```

The latest applicable trusted decision, ordered by comment update time and then
comment ID, controls the result. Editing a decision is a new verification event.
Only the current head commit is covered. New commits require a fresh verification;
this does not necessarily require a new agreement if the existing acceptance
already covers the changes. Deleting or editing away a decision removes that
decision; use an explicit revocation when earlier verification comments remain.

## Automated check and deployment

The `Contributor agreement` status verifies the PR acceptance checkbox and a
trusted maintainer's current-commit verification. It uses GitHub's authenticated
comment author and repository permissions. It does not inspect private records,
prove legal ownership, assess contractual enforceability or replace review.

The workflow runs trusted default-branch code and never checks out or executes PR
code. It reads the PR, comments and permissions, then publishes a commit status.
It does not send messages, collect identities or modify the acceptance records.
Use its manual workflow dispatch with the PR number to recheck after a permission
change or another event that did not trigger a run. Revocation affects the public
verification status; it does not itself withdraw a validly granted legal licence.
If GitHub's API or status publication fails, a previous status may remain visible,
especially for comment or manual runs that cannot discover the current commit.
Check the workflow run's health and rerun failed verification before merging;
do not rely on an old successful status during an outage.

The workflow must first be merged into the default branch, and the repository's
Actions policy must permit its triggers and token permissions. After deployment,
configure `Contributor agreement` as a required status check for `main` and review
any administrator or other bypass permissions. A workflow file alone does not
make its status a merge requirement; repository settings must require it.
Until that protection is active, maintainers must enforce the record requirement
manually. Test with a real PR before relying on the automated merge gate.

Publishing the agreement and workflow does not itself establish a recipient
statement or create contributor acceptance. Earlier contributions keep their existing rights basis;
missing commercial rights require an additional valid grant.
