const assert = require('node:assert/strict');
const test = require('node:test');
const check = require('./check-contributor-agreement.cjs');

const SHA = 'a'.repeat(40);
const OLD_SHA = 'b'.repeat(40);
const STATEMENT = 'I have read and accept the TopoBench Contributor Agreement 1.0 and the recipient statement supplied to me. I confirm authority to grant its rights for my contributions.';
const CHECKED = `- [x] ${STATEMENT}`;

function decision(body, id = 1, extra = {}) {
  return { id, body, user: { login: 'reviewer' }, updated_at: `2026-10-12T10:00:${String(id).padStart(2, '0')}Z`, ...extra };
}

function fixture(options = {}) {
  const pr = { number: 429, state: 'open', body: CHECKED, head: { sha: SHA }, base: { ref: 'main' }, ...options.pr };
  const statuses = [];
  const failures = [];
  const permissions = [];
  let reads = 0;
  const github = {
    rest: {
      pulls: { get: async () => {
        if (options.pullError) throw options.pullError;
        return { data: ++reads > 1 && options.changedPr ? options.changedPr : pr };
      } },
      issues: { listComments() {} },
      repos: {
        createCommitStatus: async (status) => { statuses.push(status); },
        getCollaboratorPermissionLevel: async ({ username }) => {
          permissions.push(username);
          if (options.permissionError) throw options.permissionError;
          return { data: { permission: options.permission || 'write' } };
        },
      },
    },
    paginate: async (method, args) => {
      assert.equal(method, github.rest.issues.listComments);
      assert.equal(args.per_page, 100);
      if (options.commentError) throw options.commentError;
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

test('dispatch validates the PR number as data and ignores other target branches', async () => {
  assert.equal((await state({ context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429' } } } })).statuses.at(-1)?.state, 'success');
  const invalid = await state({ context: { eventName: 'workflow_dispatch', payload: { inputs: { pull_request_number: '429; echo secret' } } } });
  assert.equal(invalid.statuses.length, 0);
  assert.equal(invalid.failures.length, 1);
  assert.equal((await state({ pr: { base: { ref: 'other' } } })).statuses.length, 0);
});
