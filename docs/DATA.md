# Data acquisition

Inputs are third-party public research datasets, not bundled with this repository.
Obtain them from their publishers and review the applicable terms before use.

Required layout under `data/raw` (or the shell variable `DRP_RAW_DIR`):

```text
ctrpv2/CTRPv2.0_2015_ctd2_ExpandedDataset.zip
depmap/Model.csv
depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv
```

- CTRPv2 archive: <https://d1hp0ufb0xxisr.cloudfront.net/CTD2/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset.zip>
  from the NCI CTD2/Broad resource. The preparation script checks the SHA-256 below
  and the publisher's member MD5 manifest before reading tables.
- DepMap: <https://depmap.org/portal/download/all/>. Obtain `Model.csv` and
  `OmicsExpressionProteinCodingGenesTPMLogp1.csv` with matching `ModelID` rows.
  The precise original release label was not preserved in the exported manifest;
  the hashes below identify the input bytes. Do not claim exact reproduction from
  a different release. If these exact files are unavailable, report the substituted
  release and expect cohort/score changes.

Historical input SHA-256:

| Input | SHA-256 |
| --- | --- |
| CTRPv2 ZIP | `8f62b3b5ed70cfd367cf52ce0a99884dd0a674d1a8c301474b707648689bdee3` |
| Model.csv | `b7a0c1385e6cef30132b56aff61f1261d11e3f490490b355c430d32ee0dbdcfa` |
| RNA CSV | `2a71dc94110efcc0221eae821bb93a9f03b54bea16f005818911a09d33383d56` |

The full original input/member/output checksums and cohort counts are in
`results/preparation_manifest.json`. New preparation creates its own manifest.
The script does not require DepMap inputs to match the historical hashes, allowing
explicitly documented independent reruns. Raw and processed data stay ignored.

CTD2 investigative reuse is not an assertion of commercial clearance. Check the
DepMap release's data terms as well. No raw datasets, expression matrices or
individual cell-line predictions are redistributed by this code release.