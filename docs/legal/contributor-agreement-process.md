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

Keep the actual statement in private storage outside the repository and its
worktrees. Give each contributor the complete statement and exact agreement
before asking them to accept. Identifying recipients is separate from accepting
on behalf of a contributor; publishing the project documents creates no assent.

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

Use the [acceptance form](templates/acceptance-form.txt) and
[record template](templates/acceptance-record.json) to prepare a private packet.
The templates are blank and do not constitute acceptance or verification.
For each contribution:

1. Create a private folder with a random opaque record ID. Copy the exact
   agreement, research licence and recipient statement into it; record their
   versions and SHA-256 hashes. Never put identities into the record ID.
2. Supply the complete documents to the contributor through a private channel.
   Keep delivery evidence and the actual date supplied.
3. Retain the contributor's original acceptance and authority evidence. For
   email acceptance, preserve the complete message with its headers and
   attachments. A maintainer must not manufacture or backdate acceptance.
4. Complete the record's contribution scope and evidence references. Keep its
   state as `draft`, `supplied` or `acceptance_received` until a maintainer checks
   the evidence. A template, hash or checked PR box alone does not prove assent.
5. Append verification decisions, including the reviewer, date, covered PR head
   and reason. Keep earlier evidence and decisions when correcting or extending
   a record. Only an actual review of complete records can mark it `verified`.

Keep a private index of record IDs, packet locations and states. Restrict access
to authorised record custodians, use encrypted storage and backups, and test
recovery. Folder permissions alone do not provide encryption or backup. Review
retention needs periodically under applicable law while preserving the evidence
needed for grants the project relies on. Supply parties their applicable records
through a private channel; do not publish the private index or packets.

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

The `Contributor agreement` status verifies PR acceptance checkboxes and trusted
maintainers' current-commit verification. GitHub statuses belong to commits, so
all open PRs targeting `main` with the same head commit must pass before that
commit receives success. Every run reconciles all open PRs to `main`; closing,
retargeting or updating one PR can change which others share its status.
When a PR blocks a shared commit, the failure description identifies its number.
The workflow serializes reconciliation runs and checks for PR changes before
publishing success. It uses GitHub's authenticated comment authors and repository
permissions. It does not inspect private records, prove legal ownership, assess
contractual enforceability or replace review.

Checks are asynchronous: a comment edited after it was read is checked by the
next queued run. Unchanged results for unrelated commits are not republished.
A run can report failure for one commit and success for another; inspect the
status on the PR's current head and the run's error messages.

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

### Enabling protection after deployment

1. Confirm the reviewed workflow and checker are present on `main`. A workflow
   introduced in an unmerged PR cannot be dispatched from the default branch.
2. Dispatch `contributor-agreement.yml` on `main`, supplying an open PR number.
   Confirm a fresh `Contributor agreement` result on the expected commit and
   inspect the linked workflow run. An unacknowledged or unverified PR must fail.
3. Re-read the existing branch protection. Add `Contributor agreement` to the
   required checks, binding its source to the GitHub Actions app. Resolve the
   app ID through `GET /apps/github-actions`; do not use a bot user ID. Preserve
   existing required checks and the current strict/up-to-date setting.
4. Use the targeted `PATCH /repos/{owner}/{repo}/branches/main/protection/required_status_checks`
   endpoint rather than replacing the full branch-protection configuration.
   This requires an authorised administrator; the checker's `GITHUB_TOKEN`
   is not an administration token. Read back the settings and confirm other
   review requirements, restrictions and bypass settings were preserved.
5. Inspect an unverified PR's required-check display without trying to merge
   it as a test. Confirm the new requirement is active. Existing administrator
   or role bypasses remain unless separately changed; required status checks
   do not themselves remove those bypasses.
6. Test success only with a genuinely accepted contribution and reviewed
   evidence. Confirm success after the real maintainer verification, failure
   after revocation, and fresh verification after a new commit. Synthetic
   unit-test fixtures are not production acceptance records. Report positive
   live testing as pending until valid records exist; do not manufacture a
   contributor declaration or record to make the check pass.

Reference: [GitHub status-check protection API](https://docs.github.com/en/rest/branches/branch-protection#update-status-check-protection).

Publishing the agreement and workflow does not itself establish a recipient
statement or create contributor acceptance. Earlier contributions keep their existing rights basis;
missing commercial rights require an additional valid grant.
