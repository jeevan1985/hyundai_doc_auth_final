# hyundai_document_authenticator

Production-ready toolkit for TIF document image search, photo extraction, and similarity analysis.

## Refactor Notice: Directory naming corrected from `/appdat` to `/appdata`

A project-wide refactor corrected a long-standing typo in the canonical data directory name.

- Canonical path is now: `/appdata`
- Purpose: persistent application data (e.g., FAISS indices, caches, logs)

### Replacement rules applied across the codebase

- Replace every instance of `/appdat` with `/appdata`.
- Replace every instance of `appdat/` with `appdata/`.
- Replace string literals or paths "appdat" with "appdata".
- Update variable and environment names (lowercase forms only) accordingly, e.g.:
  - `APPDAT_PATH` → `APPDATA_PATH`
  - `DEFAULT_APPDAT_DIR` → `DEFAULT_APPDATA_DIR`

Safety constraints observed:
- Do not modify unrelated words that merely contain "dat" (e.g., `database`, `data`).
- Maintain correct casing — only lowercase "appdat" forms were refactored.

### Validation checklist for maintainers

- Search for any lingering references (case-sensitive):
  - In repo root:
    - Linux/macOS: `grep -R "appdat" -n -- . | grep -v "\.git/"`
    - Windows PowerShell: `Get-ChildItem -Recurse -File | Select-String -Pattern "appdat"`
  - Expected result: zero matches.
- Run linting and unit/integration tests to confirm no regressions.

### Impact on operations

- Docker volume mounts and internal paths should reference `/appdata` going forward.
- Any user-managed configuration that previously referenced `/appdat` should be updated to `/appdata`.
- If you maintain custom environment files, rename variables that used the `APPDAT_` pattern to `APPDATA_`.

No end-user action is required if you are using the latest compose files, images, and templates included with this repository. If you copied older configuration files, update them per the rules above.
