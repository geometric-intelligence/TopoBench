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
  let sha;
  const status = async (state, description) => {
    await github.rest.repos.createCommitStatus({
      ...repository, sha, state, description, context: STATUS,
      target_url: `${process.env.GITHUB_SERVER_URL || 'https://github.com'}/${repository.owner}/${repository.repo}/actions/runs/${context.runId}`,
    });
    if (state === 'failure' || state === 'error') core.setFailed(description);
    return state;
  };

  try {
    const payload = context.payload;
    if (context.eventName === 'issue_comment' && !payload.issue?.pull_request) return;
    const value = context.eventName === 'workflow_dispatch' ? payload.inputs?.pull_request_number
      : payload.pull_request?.number || payload.issue?.number;
    if (!/^[1-9]\d*$/.test(String(value)) || !Number.isSafeInteger(Number(value))) throw new Error('Invalid PR number');
    const pull_number = Number(value);
    const eventSha = context.eventName === 'pull_request_target' && payload.pull_request?.head?.sha;
    if (/^[a-fA-F0-9]{40}$/.test(eventSha || '')) {
      sha = eventSha;
      await status('pending', 'Checking acknowledgement and maintainer record verification.');
    }
    const { data: pr } = await github.rest.pulls.get({ ...repository, pull_number });
    if (pr.base.ref !== 'main' || pr.state !== 'open') return;
    const alreadyPending = sha === pr.head.sha;
    sha = pr.head.sha;
    if (!alreadyPending) await status('pending', 'Checking acknowledgement and maintainer record verification.');
    if (!hasAcceptance(pr.body)) return await status('failure', 'Check the exact contributor agreement acknowledgement in the PR description.');

    const comments = await github.paginate(github.rest.issues.listComments, { ...repository, issue_number: pull_number, per_page: 100 });
    const decisions = comments.map((comment) => parseDecision(comment, sha)).filter(Boolean)
      .sort((a, b) => Date.parse(b.comment.updated_at) - Date.parse(a.comment.updated_at) || b.comment.id - a.comment.id);
    const permissions = new Map();
    let selected;
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
      if (permissions.get(username)) { selected = decision; break; }
    }
    if (!selected) return await status('failure', 'A maintainer must verify the contributor agreement record for this commit.');
    if (!selected.verified) return await status('failure', 'Maintainer verification was revoked for this commit.');

    const { data: current } = await github.rest.pulls.get({ ...repository, pull_number });
    if (current.head.sha !== sha || current.body !== pr.body) {
      return await status('failure', 'The pull request changed during verification; rerun the check.');
    }
    // The private record holds the actual assent and authority evidence. Neither
    // a checkbox nor this status independently establishes a contributor's rights.
    return await status('success', 'Acknowledgement present; maintainer verified the record for this commit.');
  } catch {
    const message = 'Unable to verify contributor agreement records; a maintainer must rerun the check.';
    if (sha) {
      try { await status('error', message); } catch { /* Report failure even if status publishing fails. */ }
    }
    core.setFailed(message);
  }
};
