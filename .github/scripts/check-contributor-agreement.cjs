const STATEMENT = 'I have read and accept the TopoBench Contributor Agreement 1.0 and the recipient statement supplied to me. I confirm authority to grant its rights for my contributions.';
const STATUS = 'Contributor agreement';

function hasAcceptance(body) {
  let fence = null;
  let htmlCode = null;
  const statements = [];
  // Examples and HTML comments are not an administrative acknowledgement.
  for (const line of (body || '').replace(/<!--[\s\S]*?(?:-->|$)/g, '').split(/\r?\n/)) {
    if (!fence) {
      const wasHtmlCode = Boolean(htmlCode);
      const tags = [...line.matchAll(/<(\/?)(pre|code|script|style|textarea)(?=[\s>]|$)/gi)];
      for (const tag of tags) {
        if (!htmlCode && !tag[1]) htmlCode = tag[2].toLowerCase();
        else if (tag[1] && htmlCode === tag[2].toLowerCase()) htmlCode = null;
      }
      if (wasHtmlCode || tags.length) continue;
    }
    const boundary = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (boundary) {
      if (!fence) fence = boundary[1];
      else if (boundary[1][0] === fence[0] && boundary[1].length >= fence.length && !boundary[2].trim()) fence = null;
      continue;
    }
    if (fence) continue;
    const checkbox = line.trimEnd().match(/^- \[([ xX])\] (.+)$/);
    if (checkbox && checkbox[2] === STATEMENT) statements.push(checkbox[1].toLowerCase() === 'x');
  }
  return statements.length === 1 && statements[0];
}

function parseDecision(comment, sha) {
  const marker = (comment.body || '').trim().match(/^(?:CLA-VERIFIED: ([a-fA-F0-9]{40}) ([A-Za-z0-9][A-Za-z0-9._-]{0,79})|CLA-REVOKED: ([a-fA-F0-9]{40}))$/);
  if (!marker || (marker[1] || marker[3]).toLowerCase() !== sha.toLowerCase()) return null;
  if (marker[2] && /^(?:todo|tbd|placeholder|(?:opaque[-_]|your[-_])?record[-_]?id)$/i.test(marker[2])) return null;
  return { comment, verified: Boolean(marker[1]) };
}

module.exports = async function checkContributorAgreement({ github, context, core }) {
  const repository = context.repo;
  const knownHeads = new Set();
  const pendingHeads = new Set();
  const unavailable = 'Unable to verify contributor agreement records; a maintainer must rerun the check.';
  const status = async (sha, state, description) => {
    knownHeads.add(sha);
    if (state !== 'pending') {
      try {
        const [previous] = await github.paginate(github.rest.repos.listCommitStatusesForRef,
          { ...repository, ref: sha, per_page: 100 }, ({ data }, done) => {
            const latest = data.find((item) => item.context.toLowerCase() === STATUS.toLowerCase());
            if (latest) done();
            return latest ? [latest] : [];
          });
        // GitHub caps a SHA/context at 1000 statuses. Unrelated PR events must
        // not spend that quota by repeating an unchanged result.
        if (previous?.creator?.login === 'github-actions[bot]' && previous.state === state && previous.description === description) {
          if (state === 'failure' || state === 'error') core.setFailed(description);
          return state;
        }
      } catch { /* A failed deduplication read must not prevent publication. */ }
    }
    const request = {
      ...repository, sha, state, description, context: STATUS,
      target_url: `${process.env.GITHUB_SERVER_URL || 'https://github.com'}/${repository.owner}/${repository.repo}/actions/runs/${context.runId}`,
    };
    try {
      await github.rest.repos.createCommitStatus(request);
    } catch {
      core.setFailed(unavailable);
      // An exhausted or unavailable SHA must not stop all other PRs being checked.
      if (state !== 'error') {
        try { await github.rest.repos.createCommitStatus({ ...request, state: 'error', description: unavailable }); } catch { /* Keep the run failed. */ }
      }
      return 'error';
    }
    if (state === 'failure' || state === 'error') core.setFailed(description);
    return state;
  };
  const pending = async (sha) => {
    if (!pendingHeads.has(sha)) {
      await status(sha, 'pending', 'Checking acknowledgement and maintainer record verification.');
      pendingHeads.add(sha);
    }
  };
  const openPulls = () => github.paginate(github.rest.pulls.list, {
    ...repository, state: 'open', base: 'main', per_page: 100,
  });
  const byHead = (prs) => {
    const groups = new Map();
    for (const pr of prs) {
      if (pr.state !== 'open' || pr.base.ref !== 'main') continue;
      knownHeads.add(pr.head.sha);
      const group = groups.get(pr.head.sha) || [];
      group.push(pr);
      groups.set(pr.head.sha, group);
    }
    return groups;
  };
  const snapshot = (prs = []) => JSON.stringify(prs
    .map(({ number, body }) => [number, body])
    .sort((a, b) => a[0] - b[0]));
  const permissions = new Map();
  const verify = async (pr) => {
    if (!hasAcceptance(pr.body)) return 'Check the exact contributor agreement acknowledgement in the PR description.';
    const comments = await github.paginate(github.rest.issues.listComments, { ...repository, issue_number: pr.number, per_page: 100 });
    const decisions = comments.map((comment) => parseDecision(comment, pr.head.sha)).filter(Boolean)
      .sort((a, b) => Date.parse(b.comment.updated_at) - Date.parse(a.comment.updated_at) || b.comment.id - a.comment.id);
    for (const decision of decisions) {
      const username = decision.comment.user?.login;
      if (!username) continue;
      if (!permissions.has(username)) {
        try {
          const { data } = await github.rest.repos.getCollaboratorPermissionLevel({ ...repository, username });
          permissions.set(username, ['admin', 'maintain', 'write'].includes(data.permission));
        } catch (error) {
          if (error.status !== 404) throw error;
          permissions.set(username, false);
        }
      }
      if (permissions.get(username)) {
        return decision.verified ? null : 'Maintainer verification was revoked for this commit.';
      }
    }
    return 'A maintainer must verify the contributor agreement record for this commit.';
  };

  try {
    const payload = context.payload;
    if (context.eventName === 'issue_comment' && !payload.issue?.pull_request) return;
    const value = context.eventName === 'workflow_dispatch' ? payload.inputs?.pull_request_number
      : payload.pull_request?.number || payload.issue?.number;
    if (!/^[1-9]\d*$/.test(String(value)) || !Number.isSafeInteger(Number(value))) throw new Error('Invalid PR number');
    const pull_number = Number(value);
    const eventSha = context.eventName === 'pull_request_target' && payload.pull_request?.head?.sha;
    if (/^[a-fA-F0-9]{40}$/.test(eventSha || '')) await pending(eventSha);
    const { data: pr } = await github.rest.pulls.get({ ...repository, pull_number });
    knownHeads.add(pr.head.sha);
    if (pr.base.ref === 'main' && pr.state === 'open') await pending(pr.head.sha);

    // Commit statuses belong to a SHA, not a PR. Reconcile the whole repository:
    // GitHub may replace queued concurrency jobs for unrelated PR events.
    const groups = byHead(await openPulls());
    const results = new Map();
    for (const [sha, prs] of groups) {
      for (const candidate of prs) {
        const failure = await verify(candidate);
        if (failure) { results.set(sha, `PR #${candidate.number}: ${failure}`); break; }
      }
    }

    const current = byHead(await openPulls());
    for (const sha of knownHeads) {
      if (snapshot(groups.get(sha)) !== snapshot(current.get(sha))) {
        await status(sha, 'failure', 'The PRs sharing this commit changed during verification; rerun the check.');
      } else if (!current.has(sha)) {
        await status(sha, 'failure', 'No open pull request to main is associated with this commit.');
      } else if (results.has(sha)) {
        await status(sha, 'failure', results.get(sha));
      } else {
        // Private records hold the actual assent and authority evidence.
        await status(sha, 'success', 'Every open PR to main on this commit has acknowledgement and maintainer record verification.');
      }
    }
  } catch {
    for (const sha of knownHeads) {
      await status(sha, 'error', unavailable);
    }
    core.setFailed(unavailable);
  }
};
