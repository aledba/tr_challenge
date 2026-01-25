# Legal Procedural Posture Ontology

**Thomson Reuters Data Science Challenge**  
**Domain Model for Judicial Opinion Classification**

---

## 1. Executive Summary

This document presents a **legal domain ontology** for the 224 procedural posture labels in the TR dataset. The key insight is that these labels are **NOT hierarchically related via IS-A inheritance**, but rather represent **orthogonal facets** of a case's procedural state that combine compositionally.

### Key Ontological Insight

```
A document labeled ["On Appeal", "Motion to Post Bond"] does NOT mean:
  ❌ "Motion to Post Bond" IS-A "On Appeal"

It MEANS:
  ✅ The case is IN THE CONTEXT OF "On Appeal" (Stage)
  ✅ AND the specific action is "Motion to Post Bond" (Motion)
```

---

## 1.1 Diagram Conventions

### UML Relationships (Mermaid syntax)

```mermaid
classDiagram
    direction LR
    class A
    class B
    class C
    class D
    class E
    class F
    
    A <|-- B : IS-A inheritance
    C ..> D : DEPENDS-ON
    E o-- F : HAS aggregation
```

| Syntax | Arrow | Meaning |
|--------|-------|---------|
| `<\|--` | ◁── solid + closed triangle | IS-A (inheritance) |
| `..>` | ←┄┄ dashed + open arrow | DEPENDS-ON |
| `o--` | ◇── solid + open diamond | HAS (aggregation) |
| `-->` | ←── solid + open arrow | Association |

### Data vs. Ontology
| Marker | Meaning |
|--------|---------|
| `<<abstract>>` | **My organizational category** - NOT in the data |
| `<<posture>>` | **ACTUAL label from the dataset** (224 total) |

---

## 2. The Multi-Dimensional Posture Model

The posture labels encode **multiple orthogonal dimensions**:

```mermaid
---
title: Procedural Posture Ontology - Core Dimensions
---
classDiagram
    direction TB
    
    class ProceduralPosture {
        <<abstract>>
        +String label
        +getOntologicalType()
    }
    
    class Stage {
        <<enumeration>>
        Describes WHERE in litigation lifecycle
    }
    
    class MotionType {
        <<enumeration>>
        Describes WHAT action is being requested
    }
    
    class CaseCharacteristic {
        <<enumeration>>
        Describes special ATTRIBUTES of the case
    }
    
    class ProceduralEvent {
        <<enumeration>>
        Describes WHAT HAPPENED procedurally
    }
    
    class Proceeding {
        <<enumeration>>
        Describes TYPE of legal proceeding
    }
    
    ProceduralPosture <|-- Stage
    ProceduralPosture <|-- MotionType
    ProceduralPosture <|-- CaseCharacteristic
    ProceduralPosture <|-- ProceduralEvent
    ProceduralPosture <|-- Proceeding
    
    note for Stage "Examples: On Appeal, Trial, Pre-Trial"
    note for MotionType "Examples: Motion to Dismiss, Motion for Summary Judgment"
    note for CaseCharacteristic "Examples: Consolidated Cases, Class Action"
```

---

## 3. Stage Dimension (Litigation Lifecycle)

The **Stage** dimension represents where the case is in the litigation lifecycle.

```mermaid
---
title: Stage Ontology - Litigation Lifecycle
---
classDiagram
    direction LR
    
    class Stage {
        <<abstract>>
        MY CATEGORY
    }
    
    class AppellateStage {
        <<abstract>>
        MY CATEGORY
    }
    
    class AdministrativeStage {
        <<abstract>>
        MY CATEGORY
    }
    
    Stage <|-- AppellateStage
    Stage <|-- AdministrativeStage
    
    class On_Appeal {
        <<posture>>
        count: 9197
    }
    
    class Appellate_Review {
        <<posture>>
        count: 4652
    }
    
    class Review_of_Administrative_Decision {
        <<posture>>
        count: 2773
    }
    
    class Certified_Question {
        <<posture>>
        count: 72
    }
    
    AppellateStage o-- On_Appeal
    AppellateStage o-- Appellate_Review
    AppellateStage o-- Certified_Question
    AdministrativeStage o-- Review_of_Administrative_Decision
```

**Actual posture names in data:**
- "On Appeal" (9,197)
- "Appellate Review" (4,652)
- "Review of Administrative Decision" (2,773)
- "Certified Question" (72)

---

## 4. Motion Type Dimension (Procedural Actions)

**Motions** are formal requests to the court for specific rulings. They represent ACTIONS, not states.

### 4.1 Dismissal Motions

```mermaid
---
title: Motion to Dismiss - Legal Hierarchy
---
classDiagram
    direction TB
    
    class DismissalMotion {
        <<abstract>>
        MY CATEGORY - not in data
    }
    
    class Motion_to_Dismiss {
        <<posture>>
        count: 1679
    }
    
    class Motion_to_Dismiss_for_Lack_of_SMJ {
        <<posture>>
        count: 343
    }
    
    class Motion_to_Dismiss_for_Lack_of_PJ {
        <<posture>>
        count: 204
    }
    
    class Motion_to_Dismiss_for_Lack_of_Standing {
        <<posture>>
        count: 137
    }
    
    class Motion_to_Dismiss_for_Lack_of_Jurisdiction {
        <<posture>>
        count: 124
    }
    
    DismissalMotion o-- Motion_to_Dismiss
    
    Motion_to_Dismiss <|-- Motion_to_Dismiss_for_Lack_of_SMJ
    Motion_to_Dismiss <|-- Motion_to_Dismiss_for_Lack_of_PJ
    Motion_to_Dismiss <|-- Motion_to_Dismiss_for_Lack_of_Standing
    Motion_to_Dismiss <|-- Motion_to_Dismiss_for_Lack_of_Jurisdiction
```

**Actual posture names in data:**
- "Motion to Dismiss" (1,679)
- "Motion to Dismiss for Lack of Subject Matter Jurisdiction" (343)
- "Motion to Dismiss for Lack of Personal Jurisdiction" (204)
- "Motion to Dismiss for Lack of Standing" (137)
- "Motion to Dismiss for Lack of Jurisdiction" (124)

### 4.2 Trial Phase Motions

```mermaid
---
title: Trial Phase Motions
---
classDiagram
    direction TB
    
    class TrialMotion {
        <<abstract>>
        MY CATEGORY
    }
    
    class PreTrialMotion {
        <<abstract>>
        MY CATEGORY
    }
    
    class DuringTrialMotion {
        <<abstract>>
        MY CATEGORY
    }
    
    class PostTrialMotion {
        <<abstract>>
        MY CATEGORY
    }
    
    TrialMotion <|-- PreTrialMotion
    TrialMotion <|-- DuringTrialMotion
    TrialMotion <|-- PostTrialMotion
    
    class Motion_in_Limine {
        <<posture>>
        count: 70
    }
    
    class Jury_Selection_Challenge_or_Motion {
        <<posture>>
        count: 84
    }
    
    class Motion_for_JMOL_Directed_Verdict {
        <<posture>>
        count: 212
    }
    
    class Motion_for_New_Trial {
        <<posture>>
        count: 226
    }
    
    class Motion_for_Reconsideration {
        <<posture>>
        count: 206
    }
    
    class Post_Trial_Hearing_Motion {
        <<posture>>
        count: 512
    }
    
    PreTrialMotion o-- Motion_in_Limine
    DuringTrialMotion o-- Jury_Selection_Challenge_or_Motion
    DuringTrialMotion o-- Motion_for_JMOL_Directed_Verdict
    PostTrialMotion o-- Motion_for_New_Trial
    PostTrialMotion o-- Motion_for_Reconsideration
    PostTrialMotion o-- Post_Trial_Hearing_Motion
```

### 4.3 Appellate-Specific Motions (DEPENDENCY on Stage)

```mermaid
---
title: Appellate Motions - Mixed Relationships
---
classDiagram
    direction TB
    
    class On_Appeal {
        <<posture>>
        count: 9197
    }
    
    class AppellateMotion {
        <<abstract>>
        MY CATEGORY
    }
    
    class Motion_to_Post_Bond {
        <<posture>>
        count: 2
    }
    
    class Motion_for_Appeal_Bond {
        <<posture>>
        count: 2
    }
    
    class Motion_to_Expand_the_Record {
        <<posture>>
        count: 2
    }
    
    class Motion_to_Supplement_the_Record {
        <<posture>>
        count: 20
    }
    
    class Motion_for_Rehearing {
        <<posture>>
        count: 48
    }
    
    class Motion_to_Reargue {
        <<posture>>
        count: 35
    }
    
    class Petition_for_Rehearing_En_Banc {
        <<posture>>
        count: 6
    }
    
    AppellateMotion ..> On_Appeal : depends on
    
    AppellateMotion o-- Motion_to_Post_Bond
    AppellateMotion o-- Motion_for_Appeal_Bond
    AppellateMotion o-- Motion_to_Expand_the_Record
    AppellateMotion o-- Motion_to_Supplement_the_Record
    AppellateMotion o-- Motion_for_Rehearing
    AppellateMotion o-- Motion_to_Reargue
    AppellateMotion o-- Petition_for_Rehearing_En_Banc
```

**Actual posture names in data:**
- "On Appeal" (9,197)
- "Motion to Post Bond" (2)
- "Motion for Appeal Bond" (2)
- "Motion to Expand the Record" (2)
- "Motion to Supplement the Record" (20)
- "Motion for Rehearing" (48)
- "Motion to Reargue" (35)
- "Petition for Rehearing En Banc" (6)

### 4.4 Injunction Motions

```mermaid
---
title: Injunction Motion Hierarchy
---
classDiagram
    direction LR
    
    class InjunctionMotion {
        <<abstract>>
        Request for court order to act/refrain
    }
    
    class MotionForPreliminaryInjunction {
        <<posture>>
        count: 364
    }
    
    class MotionForPermanentInjunction {
        <<posture>>
        count: 108
    }
    
    class MotionForTRO {
        <<posture>>
        count: implied
    }
    
    class MotionForRestrainingOrder {
        <<posture>>
        count: 59
    }
    
    InjunctionMotion o-- MotionForPreliminaryInjunction
    InjunctionMotion o-- MotionForPermanentInjunction
    InjunctionMotion o-- MotionForTRO
    InjunctionMotion o-- MotionForRestrainingOrder
```

---

## 5. Specialized Proceeding Types

### 5.1 Family Law Proceedings

```mermaid
---
title: Family Law Proceeding Ontology
---
classDiagram
    direction TB
    
    class FamilyLawProceeding {
        <<abstract>>
        Matters involving family relationships
    }
    
    class DivorceProceeding {
        <<abstract>>
        Dissolution of marriage
    }
    
    class CustodyProceeding {
        <<abstract>>
        Child custody matters
    }
    
    class SupportProceeding {
        <<abstract>>
        Financial support matters
    }
    
    class ParentalRightsProceeding {
        <<abstract>>
        Termination/establishment of parental rights
    }
    
    FamilyLawProceeding <|-- DivorceProceeding
    FamilyLawProceeding <|-- CustodyProceeding
    FamilyLawProceeding <|-- SupportProceeding
    FamilyLawProceeding <|-- ParentalRightsProceeding
    
    class PetitionForDivorce {
        <<posture>>
        count: 123
    }
    
    class PetitionForLegalSeparation {
        <<posture>>
        count: 1
    }
    
    class PetitionForCustody {
        <<posture>>
        count: 59
    }
    
    class PetitionForVisitation {
        <<posture>>
        count: 17
    }
    
    class MotionToModifyVisitation {
        <<posture>>
        count: 18
    }
    
    class PetitionToSetChildSupport {
        <<posture>>
        count: 36
    }
    
    class PetitionToEnforceChildSupport {
        <<posture>>
        count: 12
    }
    
    class MotionToModifyAlimony {
        <<posture>>
        count: 24
    }
    
    class PetitionToTerminateParentalRights {
        <<posture>>
        count: 219
    }
    
    class PetitionForAdoption {
        <<posture>>
        count: 27
    }
    
    DivorceProceeding o-- PetitionForDivorce
    DivorceProceeding o-- PetitionForLegalSeparation
    CustodyProceeding o-- PetitionForCustody
    CustodyProceeding o-- PetitionForVisitation
    CustodyProceeding o-- MotionToModifyVisitation
    SupportProceeding o-- PetitionToSetChildSupport
    SupportProceeding o-- PetitionToEnforceChildSupport
    SupportProceeding o-- MotionToModifyAlimony
    ParentalRightsProceeding o-- PetitionToTerminateParentalRights
    ParentalRightsProceeding o-- PetitionForAdoption
```

### 5.2 Criminal Proceedings

```mermaid
---
title: Criminal Proceeding Ontology
---
classDiagram
    direction TB
    
    class CriminalProceeding {
        <<abstract>>
        Criminal law matters
    }
    
    class SentencingPhase {
        <<posture>>
        count: 1342
    }
    
    class TrialPhase {
        <<posture>>
        count: 1097
    }
    
    class JuvenileProceeding {
        <<posture>>
        count: 146
    }
    
    class BailMotion {
        <<posture>>
        count: 49
    }
    
    CriminalProceeding o-- SentencingPhase
    CriminalProceeding o-- TrialPhase
    CriminalProceeding o-- JuvenileProceeding
    CriminalProceeding o-- BailMotion
```

### 5.3 Bankruptcy Proceedings

```mermaid
---
title: Bankruptcy Proceeding Ontology
---
classDiagram
    direction TB
    
    class BankruptcyProceeding {
        <<abstract>>
        Matters under Bankruptcy Code
    }
    
    class PlanRelated {
        <<abstract>>
        Chapter 11/13 plan matters
    }
    
    class ClaimsRelated {
        <<abstract>>
        Proof of claim disputes
    }
    
    class StayRelated {
        <<abstract>>
        Automatic stay matters
    }
    
    class PropertyRelated {
        <<abstract>>
        Estate property matters
    }
    
    BankruptcyProceeding <|-- PlanRelated
    BankruptcyProceeding <|-- ClaimsRelated
    BankruptcyProceeding <|-- StayRelated
    BankruptcyProceeding <|-- PropertyRelated
    
    class ObjectionToConfirmation {
        <<posture>>
        count: 27
    }
    
    class MotionToConfirmPlan {
        <<posture>>
        count: 5
    }
    
    class ObjectionToProofOfClaim {
        <<posture>>
        count: 49
    }
    
    class MotionForReliefFromStay {
        <<posture>>
        count: 18
    }
    
    class MotionToUseCashCollateral {
        <<posture>>
        count: 5
    }
    
    class MotionToConvertOrDismiss {
        <<posture>>
        count: 32
    }
    
    PlanRelated o-- ObjectionToConfirmation
    PlanRelated o-- MotionToConfirmPlan
    ClaimsRelated o-- ObjectionToProofOfClaim
    StayRelated o-- MotionForReliefFromStay
    PropertyRelated o-- MotionToUseCashCollateral
    BankruptcyProceeding o-- MotionToConvertOrDismiss
```

---

## 6. Louisiana Civil Law Exceptions

Louisiana, as a civil law jurisdiction, uses distinct procedural terminology:

```mermaid
---
title: Louisiana Exception Ontology
---
classDiagram
    direction TB
    
    class LouisianaException {
        <<abstract>>
        Louisiana Code of Civil Procedure exceptions
    }
    
    class DeclinatoryException {
        <<abstract>>
        Challenges to court's authority (Art. 921-932)
    }
    
    class DilatoryException {
        <<abstract>>
        Delays proceedings (Art. 921, 926-928)
    }
    
    class PeremptoryException {
        <<abstract>>
        Bars the action entirely (Art. 921, 927-930)
    }
    
    LouisianaException <|-- DeclinatoryException
    LouisianaException <|-- DilatoryException
    LouisianaException <|-- PeremptoryException
    
    class DecSubjectMatter {
        <<posture>>
        count: 3
    }
    
    class DecPersonalJurisdiction {
        <<posture>>
        count: 1
    }
    
    class DecImproperVenue {
        <<posture>>
        count: 1
    }
    
    class DecLisPendens {
        <<posture>>
        count: 2
    }
    
    class DecInsuffService {
        <<posture>>
        count: 1
    }
    
    class DilUnauthorizedSummary {
        <<posture>>
        count: 1
    }
    
    class PerResJudicata {
        <<posture>>
        count: 7
    }
    
    class PerNoRightOfAction {
        <<posture>>
        count: 5
    }
    
    class PerPeremption {
        <<posture>>
        count: 3
    }
    
    class PerAbandonment {
        <<posture>>
        count: 3
    }
    
    class PerNonjoinder {
        <<posture>>
        count: 1
    }
    
    DeclinatoryException o-- DecSubjectMatter
    DeclinatoryException o-- DecPersonalJurisdiction
    DeclinatoryException o-- DecImproperVenue
    DeclinatoryException o-- DecLisPendens
    DeclinatoryException o-- DecInsuffService
    DilatoryException o-- DilUnauthorizedSummary
    PeremptoryException o-- PerResJudicata
    PeremptoryException o-- PerNoRightOfAction
    PeremptoryException o-- PerPeremption
    PeremptoryException o-- PerAbandonment
    PeremptoryException o-- PerNonjoinder
```

---

## 7. Relationship Summary Diagram

```mermaid
---
title: Posture Relationship Types - Correct UML
---
classDiagram
    direction TB
    
    class Stage {
        <<Context>>
        On Appeal
        Trial
        Administrative Review
    }
    
    class Motion {
        <<Action>>
        Motion to Dismiss
        Motion for Summary Judgment
    }
    
    class AppellateMotion {
        <<Specialized Action>>
        Motion to Post Bond
        Motion to Expand Record
    }
    
    class Proceeding {
        <<Domain>>
        Family Law
        Bankruptcy
        Criminal
    }
    
    class SpecificMotion {
        <<Specialized Action>>
        MTD for Lack of SMJ
        Motion for Preliminary Injunction
    }
    
    Motion <|-- AppellateMotion
    Motion <|-- SpecificMotion
    AppellateMotion ..> Stage : depends on
    Motion --> Proceeding : categorized by
```

---

## 8. Implications for Modeling

### 8.1 Why This Matters for ML

1. **Not 224 independent labels** - Labels have structure
2. **Constraint satisfaction** - Some combinations are impossible
3. **Hierarchical loss functions** - Errors within a family are less severe
4. **Multi-task learning** - One head per dimension

### 8.2 Suggested Label Factorization

```mermaid
---
title: Suggested Multi-Task Architecture
---
classDiagram
    direction LR
    
    class Document {
        +String text
        +List~String~ postures
    }
    
    class StageClassifier {
        <<Task Head>>
        +predict() Stage
        Output: On Appeal, Trial, Admin, etc.
    }
    
    class MotionClassifier {
        <<Task Head>>
        +predict() List~Motion~
        Output: Multi-label motions
    }
    
    class DomainClassifier {
        <<Task Head>>
        +predict() Domain
        Output: Family, Bankruptcy, Criminal, Civil
    }
    
    class SharedEncoder {
        <<BERT/LegalBERT>>
        +encode() Embedding
    }
    
    Document --> SharedEncoder
    SharedEncoder --> StageClassifier
    SharedEncoder --> MotionClassifier
    SharedEncoder --> DomainClassifier
```

### 8.3 Constraint Rules (Pseudo-code)

```python
# Appellate motions require appellate stage
if "Motion to Post Bond" in predictions:
    assert "On Appeal" in predictions or "Appellate Review" in predictions

# Criminal phase motions are mutually exclusive
assert not ("Sentencing Phase" in predictions and "Trial Phase" in predictions)

# Louisiana exceptions only in Louisiana courts
if any(p.startswith("Peremptory Exception") for p in predictions):
    assert jurisdiction == "Louisiana"
```

---

## 9. Top 30 Postures Classified

| Rank | Posture | Count | Dimension | Subcategory |
|------|---------|-------|-----------|-------------|
| 1 | On Appeal | 9,197 | **Stage** | Appellate |
| 2 | Appellate Review | 4,652 | **Stage** | Appellate |
| 3 | Review of Administrative Decision | 2,773 | **Stage** | Administrative |
| 4 | Motion to Dismiss | 1,679 | **Motion** | Dismissal |
| 5 | Sentencing or Penalty Phase Motion | 1,342 | **Proceeding+Motion** | Criminal |
| 6 | Trial or Guilt Phase Motion | 1,097 | **Proceeding+Motion** | Criminal |
| 7 | Motion for Attorney's Fees | 612 | **Motion** | Fees/Costs |
| 8 | Post-Trial Hearing Motion | 512 | **Motion** | Post-Trial |
| 9 | Motion for Preliminary Injunction | 364 | **Motion** | Injunction |
| 10 | Motion to Dismiss (SMJ) | 343 | **Motion** | Dismissal |
| 11 | Motion to Compel Arbitration | 255 | **Motion** | ADR |
| 12 | Motion for New Trial | 226 | **Motion** | Post-Trial |
| 13 | Petition to Terminate Parental Rights | 219 | **Proceeding** | Family Law |
| 14 | Motion for JMOL/Directed Verdict | 212 | **Motion** | Trial |
| 15 | Motion for Reconsideration | 206 | **Motion** | Post-Ruling |
| 16 | Motion to Dismiss (Personal Jx) | 204 | **Motion** | Dismissal |
| 17 | Motion for Costs | 168 | **Motion** | Fees/Costs |
| 18 | Juvenile Delinquency Proceeding | 146 | **Proceeding** | Criminal/Juvenile |
| 19 | Motion for Default Judgment | 143 | **Motion** | Default |
| 20 | Motion to Dismiss (Standing) | 137 | **Motion** | Dismissal |
| 21 | Motion to Dismiss (Jurisdiction) | 124 | **Motion** | Dismissal |
| 22 | Motion to Transfer Venue | 124 | **Motion** | Venue |
| 23 | Petition for Divorce/Dissolution | 123 | **Proceeding** | Family Law |
| 24 | Motion for Protective Order | 116 | **Motion** | Discovery |
| 25 | Motion for Contempt | 116 | **Motion** | Enforcement |
| 26 | Motion for Permanent Injunction | 108 | **Motion** | Injunction |
| 27 | Motion to Set Aside or Vacate | 101 | **Motion** | Post-Judgment |
| 28 | Jury Selection Challenge | 84 | **Motion** | Trial |
| 29 | Motion to Renew | 74 | **Motion** | Procedural |
| 30 | Certified Question | 72 | **Stage** | Appellate |

---

## 10. Conclusion

The Thomson Reuters posture taxonomy is a **multi-faceted labeling system** that combines:

1. **Stage** (where in litigation) - mostly mutually exclusive
2. **Motion Type** (what action) - can have multiple
3. **Proceeding Type** (legal domain) - usually one
4. **Procedural Event** (what happened) - can have multiple

The **REQUIRES** relationship (e.g., Motion to Post Bond → On Appeal) is **NOT** inheritance but **contextual dependency**. This has significant implications for modeling approaches.

---

*Document generated: January 2026*  
*For Thomson Reuters Data Science Challenge*
