# Dataset access: status and request drafts (verified 2026-11-02)

| Dataset | Access | Who must act | Status |
|---|---|---|---|
| UBFC-rPPG | Public Google Drive / OneDrive, no form | nobody | **downloading to Modal volume** `/data/UBFC-rPPG` |
| SCAMPS | Public Azure blob, no form | nobody | 10 example clips + 300-clip slice on volume `/data/scamps` |
| rppg_biases | Public GitHub, MIT | nobody | on volume `/data/rppg_biases` |
| PURE | Email from an **academic address** (gmail rejected) | any team member with a university email, or supervisor | draft below, not sent |
| MMPD | Signed agreement PDF, email **must come from faculty** | supervisor | draft below, not sent |
| UBFC-Phys | IEEE DataPort "open access" (free IEEE account) or dat@UBFC server | any team member | not requested |
| COHFACE | Zenodo record 4081054 (Idiap EULA on Zenodo, login) | any team member | not requested |

Fill in: `<TEAM>`, `<UNIVERSITY>`, `<SUPERVISOR>`, `<YOUR NAME>`. Send from the university address.

---

## PURE — to `nikr-datasets-request@tu-ilmenau.de`

**Subject:** Request for PURE dataset — IEEE SP Cup 2027 student team, `<UNIVERSITY>`

Dear Prof. Gross and the NIKR dataset team,

I am `<YOUR NAME>`, an undergraduate student at `<UNIVERSITY>`, writing on behalf of our IEEE Signal Processing Cup 2027 team (team `<TEAM>`, supervised by `<SUPERVISOR>`, `<supervisor email>`). The 2027 SP Cup task is contactless estimation of heart rate, SpO2 and blood pressure from RGB facial video with explicit robustness requirements (low light, skin tone, low resolution, low frame rate, masks).

We request access to the PURE dataset (Stricker, Müller, Gross, Ro-Man 2014) to use as a reference benchmark for heart-rate and SpO2 estimation under the six motion conditions it provides. The data will be used strictly for academic research within this competition, stored on access-controlled university/cloud storage, not redistributed, and cited as requested on your dataset page. Our code is open source at https://github.com/RaihanulHaque/facevitals.

Institution: `<UNIVERSITY>`, `<department>`, `<website>`.
Supervisor: `<SUPERVISOR>`, `<title>`, `<supervisor email>`.

Thank you for making the dataset available to the research community.

Kind regards,
`<YOUR NAME>`
`<university email>` · `<UNIVERSITY>`

---

## MMPD — to `tjk24@mails.tsinghua.edu.cn`, cc `yuntaowang@tsinghua.edu.cn`
**Sender must be the faculty supervisor.** Attach the signed `MMPD_Release_Agreement.pdf` from
https://github.com/McJackTang/MMPD_rPPG_dataset. Read `Data Usage Protocol.pdf` in the same repo first.

**Subject:** MMPD Access Request - `<UNIVERSITY>`

Dear Mr. Tang and Prof. Wang,

I am `<SUPERVISOR>`, `<title>` at `<UNIVERSITY>` (`<department website>`). I supervise a student team participating in the IEEE Signal Processing Cup 2027, whose task is contactless estimation of heart rate, SpO2 and blood pressure from RGB facial video with required robustness across skin tone (Fitzpatrick I–VI), low illumination (≤ 50 lux), low resolution and low frame rate.

MMPD is the only public dataset we know of that jointly labels skin tone, lighting level, motion and exercise on mobile-phone video, which makes it the natural benchmark for the fairness and low-light requirements of the competition. We request access to the full dataset (370 GB) for academic use only.

Intended use: evaluating classical and learned rPPG methods for heart-rate estimation stratified by skin tone and lighting, and fine-tuning models for low-light robustness. No commercial use, no redistribution; results will cite Tang et al., EMBC 2023. The signed release agreement is attached.

Institution website: `<website>`. Relevant publications: `<list or "none yet; competition entry">`.
Student contact: `<YOUR NAME>`, `<university email>`.

Sincerely,
`<SUPERVISOR>`
`<title>`, `<UNIVERSITY>` · `<supervisor email>`

---

## UBFC-Phys — no email needed
1. Create a free IEEE account, open https://ieee-dataport.org/open-access/ubfc-phys-2, accept terms, download.
2. Or the French mirror: https://search-data.ubfc.fr/FR-18008901306731-2022-05-05_UBFC-Phys-A-Multimodal-Dataset-For.html
3. Upload to the volume: `conda run -n ML modal volume put facevitals-data <local dir> /UBFC-Phys`

## COHFACE — no email needed
1. Log in to https://zenodo.org/records/4081054, request access (Idiap EULA shown on the record page), download.
2. Upload: `conda run -n ML modal volume put facevitals-data <local dir> /COHFACE`

## Organizer dataset (Sri Ramachandra Hospital)
Contact Dr. Vijay Jeyakumar, `vijayjeyakumar@ieee.org`, cc `sp-competitions@listserv.ieee.org`; requires a copyright/legal agreement. Supervisor should send. Draft on request.
