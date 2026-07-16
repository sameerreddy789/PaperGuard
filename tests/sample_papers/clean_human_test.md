# Sleep Quality and Academic Performance Among First-Year College Students

## Abstract

I looked at whether self-reported sleep quality predicts GPA in a sample of 96 first-year students at a mid-sized public university. Students filled out the Pittsburgh Sleep Quality Index near the end of their first semester, and I pulled their GPA from academic records with consent. Students with poorer sleep quality tended to have lower GPAs, and the relationship held up even after I controlled for weekly study hours. The effect size was modest, but it showed up consistently across both STEM and non-STEM majors. I think this adds a small but useful data point to the growing case for colleges taking student sleep more seriously, not just as a wellness issue but as something tied to actual academic outcomes.

## Introduction

Sleep and academic performance have been linked in a number of studies, though most of that work has focused on younger students or on lab-based sleep restriction rather than the way students actually sleep in their day-to-day lives. First-year college students are an interesting group to look at because they're adjusting to a new environment, often for the first time managing their own schedule without a parent enforcing a bedtime. Walker (2017) reviewed decades of sleep research and argued that memory consolidation during sleep is one of the more well-supported mechanisms linking sleep to learning outcomes. Building on that, I wanted to see whether a simple, low-cost self-report measure of sleep quality would show a relationship with grades in a real academic setting, rather than a lab.

## Methods

I recruited 96 first-year students through a mass email sent by the registrar's office in the last two weeks of the fall semester. Participants completed the Pittsburgh Sleep Quality Index (PSQI), a validated 19-item questionnaire that produces a global score from 0 to 21, with higher scores indicating worse sleep quality. Students also reported their average weekly study hours. With consent, I obtained each participant's semester GPA from the registrar. I ran a linear regression predicting GPA from PSQI score, then a second model adding weekly study hours as a control, since study time is an obvious confound. All analyses were done in R.

## Results

The simple regression showed a negative relationship between PSQI score and GPA (b = -0.06, p = .01), meaning worse sleep quality was associated with lower grades. When I added weekly study hours to the model, the sleep effect shrank slightly but stayed statistically significant (b = -0.05, p = .03), and study hours were themselves a strong positive predictor of GPA (b = 0.04, p < .001). The overall model explained about 18% of the variance in GPA. I also checked whether the effect differed between STEM and non-STEM majors using an interaction term, and it didn't reach significance, so I'm treating the effect as fairly general across majors in this sample rather than concentrated in one group.

## Discussion

These results line up with the broader literature suggesting sleep quality matters for academic outcomes, even outside a lab setting. One thing I want to be upfront about: this is a correlational, cross-sectional study with a fairly small and single-institution sample, so I can't say sleep quality causes better grades, only that the two are related even after accounting for study time. It's also possible that some third factor, like stress or mental health, is driving both poor sleep and lower grades, and I didn't measure that here. A longitudinal design following students across their first year, with a mental health measure included, would be a natural next step. Still, given how cheap and easy the PSQI is to administer, I think there's a reasonable case for colleges building sleep screening into first-year advising rather than treating it as a separate wellness initiative.

## Conclusion

Self-reported sleep quality was associated with first-semester GPA in this sample of first-year students, even after accounting for study hours. While the design here can't establish causation, the consistency of this finding with prior research suggests sleep is worth taking seriously as an academic support issue, not just a health one.

## References

Buysse, D. J., Reynolds, C. F., Monk, T. H., Berman, S. R., & Kupfer, D. J. (1989). The Pittsburgh Sleep Quality Index: A new instrument for psychiatric practice and research. *Psychiatry Research, 28*(2), 193–213.

Walker, M. P. (2017). *Why we sleep: Unlocking the power of sleep and dreams*. Scribner.

Pilcher, J. J., & Walters, A. S. (1997). How sleep deprivation affects psychological variables related to college students' cognitive performance. *Journal of American College Health, 46*(3), 121–126.

Gaultney, J. F. (2010). The prevalence of sleep disorders in college students: Impact on academic performance. *Journal of American College Health, 59*(2), 91–97.

Trockel, M. T., Barnes, M. D., & Egget, D. L. (2000). Health-related variables and academic performance among first-year college students: Implications for sleep and other behaviors. *Journal of American College Health, 49*(3), 125–131.
