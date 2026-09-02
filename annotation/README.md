# Annotation Guidelines: Precursor Relevance

## Task Overview

For each query paper and its associated candidate papers in [``subset-evaluation.jsonl``](./subset-evaluation.jsonl), you will evaluate how strongly each candidate serves as a _precursor_ to the query paper.

A *precursor* is a paper that introduces ideas, methods, definitions, or results that are later directly used, extended, or refined in the query paper. The key question is:

> **Does the candidate paper meaningfully contribute foundational ideas that the query paper builds on?**

---

## What is a "Precursor"?

A paper is a good precursor if it satisfies at least one of the following:

- Introduces method, algorithm, or technique later used in the query paper  
- Defines concepts, frameworks, or notation adopted by the query paper  
- Including results that are extended, generalized, or refined  
- Establishes a theoretical foundation necessary for the query paper's results  
- Is explicitly referenced as an important prior work in the query paper (if known)

---

## Relevance Scale (1–10)

Assign a score from **1 to 10**, where higher values indicate stronger precursor relevance.:

#### 1–2 — No precursor relation
#### 3–4 — Weak precursor
- Indirect or minor influence  
#### 5–6 — Moderate precursor
- Contributes partial foundational support  
#### 7–8 — Strong precursor
- Important for understanding the query paper  
#### 9–10 — Essential precursor
- Foundational work  
- Core ideas, definitions, or results directly built upon  
- Critical prerequisite for the query paper  
---

## Instructions

- Focus on **directional influence**: precursor → query paper  
- Maintain consistency across all annotations   
- Save your annotation results in the same format after adding the relevance score (see example below).
- You may optionally add a note explaining your judgment.
- You are allowed to consult search engines or LLMs, but **DO NOT directly ask an LLM to assign the relevance score for you.**
  - The LLM may be used for background understanding ONLY  
  - Final scoring must be based on your own judgment  
  - **You are fully responsible for the final annotation decision**

## Annotation Example
```json
{
  "query_title": {
    "paper": "https://zbmath.org/7285677",
    "title": "Derivatives of normal functions in reverse mathematics"
  },
  "candidates": [
    {
      "paper": "https://zbmath.org/7736245",
      "title": "A note on ordinal exponentiation and derivatives of normal functions",
      "relevance": 8,
      "note": "About normal functions, important for the query paper"
    },
    {
      "paper": "https://zbmath.org/7244013",
      "title": "Computable aspects of the Bachmann-Howard principle",
      "relevance": 5,
    },
    ....
  ]
}
```
