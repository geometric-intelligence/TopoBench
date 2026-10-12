const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const check = require('./check-contributor-agreement.cjs');

const SHA = 'a'.repeat(40);
const OLD_SHA = 'b'.repeat(40);
const STATEMENT = 'I have read and accept the TopoBench Contributor Agreement 1.0 and the recipient statement supplied to me. I confirm authority to grant its rights for my contributions.';
const CHECKED = `- [x] ${STATEMENT}`;

function decision(body, id = 1, extra = {}) {
  return { id, body, user: { login: 'reviewer' }, updated_at: `2026-10-12T10:00:${String(id).padStart(2, '0')}Z`, ...extra };
}

function pull(number, sha = SHA, extra = {}) {
  return { number, state: 'open', body: CHECKED, head: { sha }, base: { ref: 'main' }, ...extra };
}

function fixture(options = {}) {
  const pr = { number: 429, state: 'open', body: CHECKED, head: { sha: SHA }, base: { ref: 'main' }, ...options.pr };
  const statuses = [];
  const failures = [];
  const permissions = [];
  const storedStatuses = options.storedStatuses || new Map();
  let reads = 0;
  let lists = 0;
  const github = {
    rest: {
      pulls: { list() {}, get: async ({ pull_number }) => {
        if (options.pullError) throw options.pullError;
        if (options.prs) return { data: options.prs.find(({ number }) => number === pull_number) || pr };
        return { data: ++reads > 1 && options.changedPr ? options.changedPr : pr };
      } },
      issues: { listComments() {} },
      repos: {
        createCommitStatus: async (status) => {
          statuses.push(status);
          if (options.statusError?.(status)) throw new Error('private status API details');
          storedStatuses.set(status.sha, { ...status, creator: { login: 'github-actions[bot]' } });
        },
        listCommitStatusesForRef() {},
        getCollaboratorPermissionLevel: async ({ username }) => {
          permissions.push(username);
          if (options.permissionError) throw options.permissionError;
          return { data: { permission: options.permission || 'write' } };
        },
      },
    },
    paginate: async (method, args, map) => {
      assert.equal(args.per_page, 100);
      if (method === github.rest.repos.listCommitStatusesForRef) {
        return map({ data: storedStatuses.has(args.ref) ? [storedStatuses.get(args.ref)] : [] }, () => {});
      }
      if (method === github.rest.pulls.list) {
        assert.equal(args.state, 'open');
        assert.equal(args.base, 'main');
        lists += 1;
        if (options.listErrorAt === lists) throw new Error('private PR discovery details');
        if (options.prSnapshots) return options.prSnapshots[Math.min(lists - 1, options.prSnapshots.length - 1)];
        const current = lists > 1 && options.changedPr ? { ...pr, ...options.changedPr } : pr;
        return (options.prs || [current]).filter(({ state, base }) => state === 'open' && base.ref === 'main');
      }
      assert.equal(method, github.rest.issues.listComments);
      if (options.commentError) throw options.commentError;
      if (options.commentsByPr) return options.commentsByPr[args.issue_number] || [];
      return options.comments || [decision(`CLA-VERIFIED: ${SHA} case.2026-001`)];
    },
  };
  const context = { repo: { owner: 'example', repo: 'topobench' }, eventName: 'pull_request_target', payload: { pull_request: { number: 429 } }, runId: 123, ...options.context };
  const core = { setFailed: (message) => failures.push(message), info() {} };
  return { github, context, core, statuses, failures, permissions };
}

async function state(options = {}) {
  const f = fixture(options);
  await check(f);
  return f;
}

test('requires a checked statement and a current-head record verified by repository permission', async () => {
  const f = await state();
  assert.deepEqual(f.statuses.map(({ state }) => state), ['pending', 'success']);
  assert.equal(f.statuses[1].sha, SHA);
  assert.equal(f.statuses[1].context, 'Contributor agreement');
  assert.deepEqual(f.permissions, ['reviewer']);
  assert.doesNotMatch(JSON.stringify(f.statuses), /reviewer|case\.2026-001/);
});

test('a verified PR cannot overwrite another PR failure on the same commit', async () => {
  const prs = [429, 430].map((number) => ({ number, state: 'open', body: CHECKED, head: { sha: SHA }, base: { ref: 'main' } }));
  for (const blocked of [
    { body: '', comments: [decision(`CLA-VERIFIED: ${SHA} fixture-record`)] },
    { body: CHECKED, comments: [] },
    { body: CHECKED, comments: [decision(`CLA-REVOKED: ${SHA}`)] },
  ]) {
    const f = await state({
      prs: [prs[0], { ...prs[1], body: blocked.body }],
      commentsByPr: { 429: [decision(`CLA-VERIFIED: ${SHA} fixture-record`)], 430: blocked.comments },
    });
    assert.equal(f.statuses.at(-1)?.state, 'failure');
    assert.equal(f.statuses.at(-1)?.sha, SHA);
    assert.ok(!f.statuses.some(({ state }) => state === 'success'));
  }
});

test('workflow serializes reconciliation and receives closing and retargeting events', () => {
  const workflow = readFileSync(path.join(__dirname, '../workflows/contributor-agreement.yml'), 'utf8');
  assert.match(workflow, /types: \[[^\]]*\bclosed\b/);
  assert.doesNotMatch(workflow, /^\s+branches:/m);
  // Job-level concurrency applies only after the non-PR issue-comment filter.
  assert.match(workflow, /^    concurrency:\n      group: [\w-]+\n      cancel-in-progress: false$/m);
  assert.doesNotMatch(workflow, /^concurrency:/m);
});

test('a status publication error fails its commit without aborting other commits', async () => {
  const otherSha = 'c'.repeat(40);
  const f = await state({
    pr: { state: 'closed' },
    prs: [SHA, OLD_SHA, otherSha].map((sha, index) => ({ number: 430 + index, state: 'open', body: CHECKED, head: { sha }, base: { ref: 'main' } })),
    commentsByPr: {
      430: [decision(`CLA-VERIFIED: ${SHA} fixture-record-a`)],
      431: [decision(`CLA-VERIFIED: ${OLD_SHA} fixture-record-b`)],
      432: [decision(`CLA-VERIFIED: ${otherSha} fixture-record-c`)],
    },
    statusError: ({ sha, state }) => sha === SHA && state === 'success',
  });
  const finalStates = new Map(f.statuses.map(({ sha, state }) => [sha, state]));
  assert.deepEqual(finalStates, new Map([[SHA, 'error'], [OLD_SHA, 'success'], [otherSha, 'success']]));
  assert.doesNotMatch(JSON.stringify(f.failures), /private status/);
});

test('manual reconciliation invalidates an orphaned commit even after its last PR closes', async () => {
  const f = await state({
    pr: { state: 'closed' },
    context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429' } } },
  });
  assert.equal(f.statuses.at(-1)?.sha, SHA);
  assert.equal(f.statuses.at(-1)?.state, 'failure');
  assert.match(f.statuses.at(-1)?.description, /No open pull request/);
});

test('unrelated events do not consume more status slots on an unchanged commit', async () => {
  const storedStatuses = new Map();
  const options = {
    storedStatuses,
    prs: [SHA, OLD_SHA].map((sha, index) => ({ number: 429 + index, state: 'open', body: CHECKED, head: { sha }, base: { ref: 'main' } })),
    commentsByPr: {
      429: [decision(`CLA-VERIFIED: ${SHA} fixture-record-a`)],
      430: [decision(`CLA-VERIFIED: ${OLD_SHA} fixture-record-b`)],
    },
  };
  await state(options);
  const second = await state(options);
  assert.equal(second.statuses.filter(({ sha }) => sha === OLD_SHA).length, 0);
  assert.equal(storedStatuses.get(OLD_SHA)?.state, 'success');
});

test('one surviving queued event reconciles other commits and keeps their outcomes separate', async () => {
  const f = await state({
    prs: [pull(429), pull(430, OLD_SHA)],
    commentsByPr: {
      429: [decision(`CLA-REVOKED: ${SHA}`)],
      430: [decision(`CLA-VERIFIED: ${OLD_SHA} fixture-record`)],
    },
    context: { eventName: 'issue_comment', payload: { issue: { number: 430, pull_request: {} } } },
  });
  assert.deepEqual(new Map(f.statuses.map(({ sha, state }) => [sha, state])), new Map([[OLD_SHA, 'success'], [SHA, 'failure']]));
});

test('closing or retargeting a blocked PR releases its verified sibling on the same commit', async () => {
  for (const departing of [{ state: 'closed' }, { base: { ref: 'other' } }]) {
    const f = await state({
      pr: departing,
      prs: [pull(430)],
      commentsByPr: { 430: [decision(`CLA-VERIFIED: ${SHA} fixture-record`)] },
      context: { payload: { pull_request: { number: 429, head: { sha: SHA } } } },
    });
    assert.equal(f.statuses.at(-1)?.state, 'success');
    assert.equal(f.statuses.at(-1)?.sha, SHA);
  }
});

test('moving one PR to a new commit rechecks both its new head and its former siblings', async () => {
  const f = await state({
    prs: [pull(429, OLD_SHA), pull(430)],
    commentsByPr: {
      429: [decision(`CLA-VERIFIED: ${SHA} fixture-stale-record`)],
      430: [decision(`CLA-VERIFIED: ${SHA} fixture-current-record`)],
    },
  });
  assert.deepEqual(new Map(f.statuses.map(({ sha, state }) => [sha, state])), new Map([[OLD_SHA, 'failure'], [SHA, 'success']]));
});

test('success requires verification for every PR even when all acknowledge the same commit', async () => {
  const f = await state({
    prs: [pull(429), pull(430)],
    commentsByPr: {
      429: [decision(`CLA-VERIFIED: ${SHA} fixture-record-a`)],
      430: [decision(`CLA-VERIFIED: ${SHA} fixture-record-b`)],
    },
  });
  assert.deepEqual(f.statuses.map(({ state }) => state), ['pending', 'success']);
});

test('new, removed, retargeted, moved, or edited group members prevent stale success', async () => {
  const original = [pull(429), pull(430)];
  for (const changed of [
    [...original, pull(431)],
    [original[0]],
    [original[0], pull(430, SHA, { base: { ref: 'other' } })],
    [original[0], pull(430, OLD_SHA)],
    [original[0], pull(430, SHA, { body: '' })],
  ]) {
    const f = await state({
      prSnapshots: [original, changed],
      commentsByPr: {
        429: [decision(`CLA-VERIFIED: ${SHA} fixture-record-a`)],
        430: [decision(`CLA-VERIFIED: ${SHA} fixture-record-b`)],
      },
    });
    assert.ok(f.statuses.some(({ sha, state }) => sha === SHA && state === 'failure'));
    assert.ok(!f.statuses.some(({ state }) => state === 'success'));
  }
});

test('discovery errors fail every known head without leaking API details', async () => {
  const prs = [pull(429), pull(430, OLD_SHA)];
  for (const listErrorAt of [1, 2]) {
    const f = await state({ prs, listErrorAt });
    const finalStates = new Map(f.statuses.map(({ sha, state }) => [sha, state]));
    assert.equal(finalStates.get(SHA), 'error');
    if (listErrorAt === 2) assert.equal(finalStates.get(OLD_SHA), 'error');
    assert.ok(!f.statuses.some(({ state }) => state === 'success'));
    assert.doesNotMatch(JSON.stringify(f.failures), /private PR discovery/);
  }
});

test('normal issue comments do not discover PRs or publish statuses', async () => {
  const f = await state({ context: { eventName: 'issue_comment', payload: { issue: { number: 123 } } } });
  assert.equal(f.statuses.length, 0);
  assert.equal(f.failures.length, 0);
});

test('checkbox alone is insufficient and unchecked assent blocks a verified record', async () => {
  for (const options of [{ comments: [] }, { pr: { body: `- [ ] ${STATEMENT}` } }]) {
    assert.equal((await state(options)).statuses.at(-1)?.state, 'failure');
  }
});

test('ignores checkbox text hidden in HTML comments, fenced code or quoted/indented examples', async () => {
  for (const body of [`<!--\n${CHECKED}\n-->`, `<!--\n${CHECKED}`, `\`\`\`markdown\n${CHECKED}\n\`\`\``, `~~~\n${CHECKED}\n~~~`, `> ${CHECKED}`, `    ${CHECKED}`]) {
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure', body);
  }
  assert.equal((await state({ pr: { body: `~~~\nexample\n~~~\n${CHECKED.replace('[x]', '[X]')}` } })).statuses.at(-1)?.state, 'success');
});

test('rejects changed wording and duplicate contradictory statements', async () => {
  for (const body of [CHECKED.replace('confirm authority', 'do not confirm authority'), `${CHECKED}\n- [ ] ${STATEMENT}`]) {
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure');
  }
});

test('raw HTML code examples cannot acknowledge the contributor agreement', async () => {
  for (const tag of ['pre', 'code', 'script', 'style', 'textarea']) {
    const body = `<${tag}>\n${CHECKED}\n</${tag}>`;
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure', tag);
    assert.equal((await state({ pr: { body: `${body}\n${CHECKED}` } })).statuses.at(-1)?.state, 'success');
  }
});

test('multiline and consecutive raw HTML opening tags remain excluded', async () => {
  for (const opening of ['<pre\nclass="example">', '<pre>example</pre><pre>']) {
    const body = `${opening}\n${CHECKED}\n</pre>`;
    assert.equal((await state({ pr: { body } })).statuses.at(-1)?.state, 'failure', opening);
  }
});

test('stale SHA and an untrusted author association cannot verify the current contribution', async () => {
  assert.equal((await state({ comments: [decision(`CLA-VERIFIED: ${OLD_SHA} case.2026-001`)] })).statuses.at(-1)?.state, 'failure');
  const f = await state({ permission: 'read', comments: [decision(`CLA-VERIFIED: ${SHA} case.2026-001`, 1, { author_association: 'OWNER' })] });
  assert.equal(f.statuses.at(-1)?.state, 'failure');
});

test('accepts only exact standalone decisions and valid non-placeholder record IDs', async () => {
  for (const body of [`Example: CLA-VERIFIED: ${SHA} case.2026-001`, `\`\`\`\nCLA-VERIFIED: ${SHA} case.2026-001\n\`\`\``, `CLA-VERIFIED: ${SHA} <record-id>`, `CLA-VERIFIED: ${SHA} TODO`, `CLA-VERIFIED: ${SHA} ${'x'.repeat(81)}`, `CLA-VERIFIED: ${SHA} a/b`]) {
    assert.equal((await state({ comments: [decision(body)] })).statuses.at(-1)?.state, 'failure', body);
  }
});

test('latest trusted current-head decision wins, including an edited older comment', async () => {
  const verified = decision(`CLA-VERIFIED: ${SHA} case.2026-001`, 1);
  const revoked = decision(`CLA-REVOKED: ${SHA}`, 2);
  assert.equal((await state({ comments: [verified, revoked] })).statuses.at(-1)?.state, 'failure');
  assert.equal((await state({ comments: [revoked, { ...verified, updated_at: '2026-10-12T11:00:00Z' }] })).statuses.at(-1)?.state, 'success');
});

test('recognizes admin and maintain repository permissions', async () => {
  for (const permission of ['admin', 'maintain']) {
    assert.equal((await state({ permission })).statuses.at(-1)?.state, 'success');
  }
});

test('deleted comments are not trusted from event payload, and all comments use pagination', async () => {
  const f = await state({ comments: [], context: { eventName: 'issue_comment', payload: { action: 'deleted', issue: { number: 429, pull_request: {} }, comment: decision(`CLA-VERIFIED: ${SHA} case.2026-001`) } } });
  assert.equal(f.statuses.at(-1)?.state, 'failure');
});

test('API failures leave an error status without publishing API details or identities', async () => {
  for (const options of [{ commentError: new Error('private identity') }, { permissionError: Object.assign(new Error('private identity'), { status: 403 }) }]) {
    const f = await state(options);
    assert.equal(f.statuses.at(-1)?.state, 'error');
    assert.doesNotMatch(JSON.stringify(f.failures), /private identity/);
  }
});

test('a permission lookup returning 404 is an untrusted commenter, not an approval', async () => {
  const f = await state({ permissionError: Object.assign(new Error('Not found'), { status: 404 }) });
  assert.equal(f.statuses.at(-1)?.state, 'failure');
});

test('initial API failure invalidates the known event head rather than preserving an old success', async () => {
  const f = await state({ pullError: new Error('private API details'), context: { payload: { pull_request: { number: 429, head: { sha: SHA } } } } });
  assert.deepEqual(f.statuses.map(({ state }) => state), ['pending', 'error']);
  assert.equal(f.statuses.at(-1).sha, SHA);
  assert.doesNotMatch(JSON.stringify(f.failures), /private API details/);
});

test('head or assent changes during verification cannot leave success', async () => {
  for (const changedPr of [{ head: { sha: OLD_SHA }, body: CHECKED }, { head: { sha: SHA }, body: '' }]) {
    assert.equal((await state({ changedPr })).statuses.at(-1)?.state, 'failure');
  }
});

test('dispatch validates the PR number as data and does not approve unrelated target branches', async () => {
  assert.equal((await state({ context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429' } } } })).statuses.at(-1)?.state, 'success');
  const invalid = await state({ context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429; echo secret' } } } });
  assert.equal(invalid.statuses.length, 0);
  assert.equal(invalid.failures.length, 1);
  assert.equal((await state({ pr: { base: { ref: 'other' } } })).statuses.at(-1)?.state, 'failure');
});
