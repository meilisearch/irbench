# Published results

One folder per run, named by run id
(`<scenario>__<engine>__<model>__<track>__<hardware profile>`), written by
`irbench publish`. Each folder holds `manifest.json`, `scores.json`, `perf.json` and
`index.json` when the run produced them. A failed run still has its manifest, so
the report can show the failure.

Only runs on a reference hardware profile are published. Results from different
profiles are never compared. The report at the repository's GitHub Pages site is
built from this folder by `irbench report --results results --out site`.
