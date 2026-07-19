# Faithfulness review

Answers generated with the off-the-shelf retriever (the fine-tuned model ranked worse — see retrieval.md).

Score each as: grounded / partially grounded / hallucinated

**Automated checks (objective, no human judgement needed):**

| Check | Result |
|---|---|
| Unanswerable questions correctly refused | **5 / 5** |
| Hallucinated answers to out-of-corpus questions | **0** |
| Citations pointing to abstracts not in retrieved context | **0 / 35** |
| Answerable questions answered | 10 / 15 |
| Refusals on answerable questions | 5 (3 correct — retrieval missed; 2 false refusals) |

The refusal path is well calibrated on out-of-corpus questions: it declined all
five, inventing nothing. Three of its five refusals on answerable questions were
correct, because retrieval had genuinely failed to surface the right abstract and
declining beats fabricating.

The two false refusals are the real defect: the correct abstract WAS retrieved and
the model still declined. At 2/15 that is a meaningful conservatism bias, and the
likeliest cause is the strict "use ONLY the abstracts below" instruction combined
with a small, fast model.

Every citation resolves to an abstract that was actually in the prompt context, so
the system is not inventing PMIDs.

**Manual grounding scores:** _still to be filled in below — the checks above verify
that citations exist and resolve, not that each claim is faithful to what the cited
abstract actually says. That judgement requires reading them._

## q001 (single)

**Q:** How do changes in motor unit discharge characteristics correlate with functional strength deficits during the two-day period following strenuous lower-limb exertion?

**Answer:** Not found in corpus.

**Retrieved:** 41359917, 41764119, 42001761, 41462500, 41359917

**Expected:** 41713462

**Score:** 

## q002 (single)

**Q:** Does performing nerve conduction studies with the wrist in a hyperextended position improve the detection of subclinical median nerve impairment?

**Answer:** Not found in corpus.

**Retrieved:** 41792064, 41432692, 41432692, 41678273, 41594340

**Expected:** 41792064

**Score:** 

## q003 (single)

**Q:** How do standardized measures of motor impairment and executive function correlate with objective spatiotemporal walking characteristics in patients with Parkinson's disease?

**Answer:** Standardized measures of motor impairment and executive function show significant correlations with spatiotemporal gait parameters in patients with Parkinson's disease (PD):

*   **Motor Impairment:** The Unified Parkinson's Disease Rating Scale, motor part (UPDRS-III) is directly related to step width in the OFF medication state [40959465]. Additionally, the Mini-Test of Balance Assessment System (Mini-BESTest) is inversely proportional to step width in both ON and OFF medication states [40959465]. Overall, gait measures show robust correlations with clinical indices of motor severity [41220871].
*   **Executive Function/Cognition:** Cognitive dysfunctions are directly related to gait instability [40959465]. Specifically, the Stroop-III test is directly proportional to stride time, stance phase, and double stance phase, and inversely proportional to cadence and swing phase in the ON medication state [40959465]. The Trail Making Test Part A (TMTB-A) is inversely proportional to step size, stride size, cadence, and speed in the ON medication state [40959465]. Furthermore, gait analysis using paradigms such as the Timed Up and Go (TUG) test—particularly the dual-task (TUGdt) paradigm—reveals kinematic parameters that serve as biomarkers for identifying cognitive states, such as distinguishing between normal cognition, subjective cognitive decline, and mild cognitive impairment in PD [42341295].

**Retrieved:** 40959465, 40926233, 40926233, 42341295, 41220871

**Expected:** 40959465

**Score:** 

## q004 (single)

**Q:** How do the preparatory weight-shifting mechanics and subsequent step execution during a pivot maneuver differ between healthy elderly individuals and patients with Parkinson's disease?

**Answer:** In patients with Parkinson's disease (pwP), anticipatory postural adjustments (APAs) during turning—specifically the unloading phase—are primarily mediolateral and exhibit a significantly reduced amplitude compared to older adults (OA) [41429909]. Furthermore, while OA demonstrate a significant association between APAs and step characteristics, this association is absent in pwP [41429909]. These impairments in APAs during turning persist in pwP even when tested while on medication [41429909].

**Retrieved:** 41429909, 41094822, 42399765, 41435746, 40959465

**Expected:** 41429909

**Score:** 

## q005 (single)

**Q:** How do corrective shoe inserts influence joint kinetics and patellofemoral pressure when athletes perform a lunging movement?

**Answer:** When athletes perform a forward lunge in badminton, the use of functional arch support insoles results in a reduced vertical ground reaction force loading rate and reduced peak negative ankle power [42004710]. However, these insoles also significantly increase peak patellofemoral joint stress and the stress-time integral [42004710].

**Retrieved:** 42004710, 41803213, 41137662, 41803213, 41090637

**Expected:** 42004710

**Score:** 

## q006 (single)

**Q:** How do knee joint movement patterns and cartilage contact mechanics differ between the reconstructed limb and the healthy limb in patients 12 to 24 months following combined ligament and meniscus repair?

**Answer:** In patients 1 to 2 years after combined anterior cruciate ligament reconstruction and meniscal surgery (ACLR+M), the reconstructed limb exhibits altered joint kinematics and cartilage contact mechanics compared to the contralateral limb during demanding activities such as landing, lunging, and pivoting [41943805]. Specifically, the ACLR+M knee demonstrates less flexion during landing and lunging, as well as increased anterior tibial translation during landing and pivoting [41943805]. Regarding cartilage contact mechanics, the reconstructed knee shows posterior shifts in contact location on the tibial cartilage surface and increased overlapping contact in both the medial and lateral compartments during landing and pivoting [41943805]. 

While gait biomechanics (such as peak knee extension angle and peak knee extension moment) in patients who underwent primary unilateral ACLR (without specific mention of combined meniscal repair) show a return to normal levels by 24 months post-operation, the knee joint's response to increased loads remains impaired for at least one year [41660297, 42452538].

**Retrieved:** 41541483, 42452538, 41943805, 41541483, 41660297

**Expected:** 41943805

**Score:** 

## q007 (single)

**Q:** Which deep learning architecture demonstrates the highest predictive accuracy for determining walking velocity when integrating kinematic, kinetic, and muscular activity data?

**Answer:** Not found in corpus.

**Retrieved:** 41171649, 40972273, 41336102, 41687508, 42435638

**Expected:** 41758888

**Score:** 

## q008 (single)

**Q:** To what extent does hip replacement surgery resolve long-term musculoskeletal and kinetic abnormalities in patients with severe congenital hip dislocation?

**Answer:** In patients with highly dislocated hips due to developmental dysplasia of the hip or sequelae of septic arthritis, total hip arthroplasty (THA) with subtrochanteric osteotomy results in improved gait parameters and clinical outcomes at one year postoperatively [41156314]. Specifically, terminal double support and vertical ground reaction forces show significant improvement at one year [41156314]. However, there is no further improvement in gait parameters at five years postoperatively compared to the one-year follow-up [41156314]. Additionally, while the mean Harris Hip Score improves significantly, and preoperative leg length discrepancy is reduced from 43.6 mm to 9.8 mm, the data indicates that gait recovery does not continue to improve beyond the first year [41156314].

**Retrieved:** 41572909, 41156314, 42114041, 41572909, 41611555

**Expected:** 41651450

**Score:** 

## q009 (single)

**Q:** How can structured semantic modeling be utilized to assist physical therapists in identifying the physiological origins of walking abnormalities in patients recovering from cerebrovascular accidents?

**Answer:** Structured semantic modeling can be utilized through the development of a knowledge graph-based cause retrieval system [41337175]. By organizing clinical knowledge—such as the Rancho Los Amigos framework—into a structured format, this system represents the relationships between gait deviations and their underlying contributors [41337175]. This allows clinicians to efficiently explore potential contributors to observed gait deviations, thereby supporting cause identification and the development of more targeted rehabilitation strategies [41337175].

**Retrieved:** 41985303, 41089595, 41337175, 41544186, 42317384

**Expected:** 41337175

**Score:** 

## q010 (single)

**Q:** How does the efficacy of treadmill-based pneumatic unloading compare to hydrotherapy or sedentary management for alleviating symptoms and enhancing mobility in patients with degenerative joint disease of the knee?

**Answer:** In a pilot randomized clinical trial involving individuals with knee osteoarthritis, lower body positive pressure (LBPP) treadmill walking demonstrated statistically significant and clinically relevant improvements in pain (KOOS-Pain) and mobility (6-minute walk test) over an eight-week period [41985188]. In contrast, participants in the aquatic pool walking group and the control (sedentary) group did not show statistically significant changes in KOOS-Pain scores [41985188]. Post-hoc analyses indicated that the improvements observed in the LBPP group were significantly larger than those in the aquatic and control groups [41985188]. Additionally, while the LBPP group showed improvements in most KOOS sub-scales, the control group experienced significant increases in IL-1β and IL-1RA serum cytokine concentrations [41985188].

**Retrieved:** 41985188, 41024213, 41140925, 41985188, 40887656

**Expected:** 41985188

**Score:** 

## q011 (single)

**Q:** How can clinicians improve the accuracy of identifying movement irregularities in individuals with degenerative hip joint disease compared to traditional single-variable analysis?

**Answer:** Clinicians can improve the assessment of movement irregularities in patients with hip osteoarthritis by using Uncontrolled Manifold (UCM) analysis, which allows for the evaluation of multiple-joint utilization to control the center of mass (CoM) during gait [42150314]. Unlike traditional indices, which do not adequately evaluate multi-joint coordination, UCM analysis provides a quantitative assessment of compensatory strategies by analyzing the variance of elemental variables (segment angles) that affect the performance variable (frontal-plane CoM) [42150314]. Additionally, instrumented 3D gait analysis can be used to provide objective data on joint angles, muscle activity, and joint loading to identify gait disturbances such as pelvic drop and lateral trunk lean [41642290].

**Retrieved:** 42150314, 42150314, 40968908, 41156314, 41642290

**Expected:** 41177154

**Score:** 

## q012 (single)

**Q:** How do the biomechanical demands and muscular recruitment patterns of reverse-stroke wheelchair propulsion compare to the conventional forward-stroke technique in patients with spinal cord injuries?

**Answer:** Compared to the conventional pushing (forward-stroke) propulsion method, pulling (reverse-stroke) propulsion results in significantly longer propulsion time and greater ranges of motion in the shoulder and elbow joints [41485206]. Regarding muscular recruitment, pulling propulsion elicits significantly higher activation of posterior upper-limb and trunk muscles, whereas pushing propulsion predominantly activates anterior shoulder muscles [41485206]. Pulling propulsion is suggested as an alternative strategy to redistribute upper limb muscular load [41485206].

**Retrieved:** 40991588, 40991588, 41485206, 41980400, 41980400

**Expected:** 41485206

**Score:** 

## q013 (single)

**Q:** How does the structural integrity and mechanical performance of a bone-patellar tendon-bone graft compare to the traditional palmaris longus tendon graft when used for medial elbow ligament repair?

**Answer:** When used for ulnar collateral ligament reconstruction, the palmaris longus graft demonstrates a higher torque at failure compared to the bone-patellar tendon-bone (BTB) graft (13.7 ± 4.9 Nm vs. 5.9 ± 1.0 Nm; P = .001) [41179402]. The BTB reconstruction exhibits a high failure rate, with 7 of 8 grafts failing during cycling, primarily at the humeral-sided bone-tendon interface [41179402]. In contrast, the palmaris longus reconstruction does not differ from the native ulnar collateral ligament in initial ligament length, elongation during cycling, stiffness, or opening angle at failure, and is considered the preferred graft choice [41179402].

**Retrieved:** 41179402, 41179402, 41582730, 41475367, 41549263

**Expected:** 41179402

**Score:** 

## q014 (single)

**Q:** Does integrating physiological constraints into vision-based pose estimation algorithms provide a statistically significant improvement in the precision of lower-limb joint angle measurements?

**Answer:** Not found in corpus.

**Retrieved:** 42356801, 41193590, 41193590, 42427911, 42427911

**Expected:** 41091616

**Score:** 

## q015 (single)

**Q:** How does the use of upper-limb support influence joint torque, postural stability, and movement fluidity during the transition between sitting and standing in individuals with degenerative joint disease?

**Answer:** Not found in corpus.

**Retrieved:** 41901033, 42235271, 41926075, 41055862, 41401653

**Expected:** 42235271

**Score:** 

## q036 (unanswerable)

**Q:** What is the recommended first-line antibiotic for uncomplicated cystitis?

**Answer:** Not found in corpus.

**Retrieved:** 41393516, 41774176, 41428956, 41267130, 41849693

**Expected:** (none — should refuse)

**Score:** 

## q037 (unanswerable)

**Q:** How does metformin lower blood glucose at the cellular level?

**Answer:** Not found in corpus.

**Retrieved:** 42208106, 41010370, 42198240, 41866084, 42242495

**Expected:** (none — should refuse)

**Score:** 

## q038 (unanswerable)

**Q:** What are the diagnostic criteria for diabetic retinopathy?

**Answer:** Not found in corpus.

**Retrieved:** 41701403, 41759651, 42138399, 42054723, 41934158

**Expected:** (none — should refuse)

**Score:** 

## q039 (unanswerable)

**Q:** Which vaccines are recommended during the first year of life?

**Answer:** Not found in corpus.

**Retrieved:** 42361683, 42361683, 41537160, 42122347, 41896993

**Expected:** (none — should refuse)

**Score:** 

## q040 (unanswerable)

**Q:** What is the mechanism of action of SSRIs in treating depression?

**Answer:** Not found in corpus.

**Retrieved:** 41027081, 42143463, 41140083, 42143463, 41687574

**Expected:** (none — should refuse)

**Score:** 
