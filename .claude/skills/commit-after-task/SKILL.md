---
name: commit-after-task
description: Commit work to git with clear Conventional Commit messages as soon as a task or feature is finished in this repo (kural-270M). Use after completing any code, config, docs or test change — a new feature, a bug fix, a refactor, a new dataset/source, a config tweak — before reporting the task as done. Also use when the user asks to "commit", "save progress", or "checkpoint" the work.
---

# Commit after every finished task

In this repository every completed task ends with a commit. Do not leave finished work
uncommitted, and do not batch unrelated tasks into one commit.

## When

- Right after a task or feature is done **and verified** (tests / the relevant smoke step ran).
- Before telling the user the task is complete — the final message should name the commit(s).
- Not in the middle of a task, and not for work that is broken: if tests fail, fix them first,
  or ask the user whether to commit anyway (then say so in the commit body).

## Steps

1. **Look before staging**
   ```bash
   git status --short
   git diff            # unstaged
   git diff --cached   # already staged (the user may have staged things themselves)
   ```
   Read what changed. Only commit changes that belong to the task you just finished; leave
   unrelated edits (e.g. the user's own work in progress) unstaged and mention them.

2. **Never commit generated artifacts or secrets.** They are git-ignored, but double-check
   nothing slipped in: `workspace/`, `wandb/`, `.venv/`, `*.safetensors`, `*.bin`, `*.pt`,
   `*.gguf`, `.env`, tokens, API keys, HF/W&B credentials, large data files. If something like
   that shows up in `git status`, stop and add it to `.gitignore` instead.

3. **Branch policy.** Don't commit straight to `main`. If the current branch is `main`, create
   a descriptive branch first and keep committing there for the rest of the work:
   `git switch -c feat/<short-topic>` (or `fix/…`, `docs/…`, `chore/…`). Never push, merge,
   rebase shared history, force-push or amend an already-pushed commit unless the user asks.

4. **Run the tests** that cover the change (`python -m pytest -q`, or the relevant smoke step).
   Mention the result in the final message.

5. **Stage by logical unit.** One commit = one coherent change that could be reverted on its
   own. A feature plus its config and tests is one commit; an unrelated docs fix is another.
   Use explicit paths: `git add <paths>`, not `git add -A` unless every change belongs together.

6. **Write the message** (Conventional Commits):

   ```
   <type>(<scope>): <imperative summary, ≤ 72 chars, no trailing period>

   <body: what changed and WHY, wrapped at ~72 chars. Mention verification
   (tests, smoke run results) and anything the reader must know.>

   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   ```

   - **type**: `feat` (new capability), `fix` (bug), `refactor` (no behaviour change),
     `perf`, `test`, `docs`, `chore` (tooling, deps, gitignore), `build`, `ci`, `data`
     (dataset sources / mixture changes), `exp` (experiment configs / results).
   - **scope** (this repo's areas): `data`, `tokenizer`, `training`, `sft`, `eval`,
     `inference`, `server`, `configs`, `scripts`, `common`, `docs`, `tests`, `deps`.
   - The summary says what the commit does, e.g. `feat(data): add Tamil SQuAD 2.0 passages
     as a pretraining source` — not `update files`, `changes`, `wip`, `fix stuff`.
   - Use the attribution trailer given by the current session's instructions; the line above is
     the current one.

   Pass multi-line messages with a heredoc (bash) or a here-string (PowerShell) so the body
   keeps its line breaks.

7. **Commit and confirm**
   ```bash
   git commit -F <message-file-or-heredoc>
   git log --oneline -n 5
   ```
   Never use `--no-verify`. If a pre-commit hook fails, fix the cause and create a new commit.

8. **Report**: in the final message list each commit as `<short-sha> <summary>` and the branch.

## Examples

```
feat(tokenizer): extend Gemma 3 vocab with Tamil BPE pieces

Adds tokenizer/adapt.py: Tamil-only pieces from a Tamil BPE are merged into
Gemma's SentencePiece model, filling the 6,242 <unusedN> slots first.
English/code encodings stay byte-identical; Tamil needs 27% fewer tokens
on the sample corpus. Verified with tests/test_tokenizer_and_training.py.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

```
fix(packing): anchor token budget on the largest mixture category

The budget was limited by the smallest category (183 tokens of spoken
Tamil), producing a tiny training set. Small categories are now capped at
max_epochs instead.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
