from SPARQLWrapper import SPARQLWrapper, JSON
import json
from pathlib import Path

# ---------------- CONFIG ----------------
endpoint_url = "http://212.227.170.235:8890/sparql"

msc_prefixes = {
    "03": "Mathematical Logic",
    "05": "Combinatorics / Graph Theory",
    "11": "Number Theory",
    "55": "Algebraic Topology",
    "60": "Probability Theory",
    "68": "Computer Science",
}

YEAR_MIN = 2020
YEAR_MAX = 2025
LIMIT_PER_MSC = 5  # how many papers to sample per MSC

# ---------------- SPARQL SETUP ----------------
sparql = SPARQLWrapper(endpoint_url)
sparql.setReturnFormat(JSON)

all_papers = []

query_template = """
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX schema: <https://schema.org/>
PREFIX msc: <http://msc2010.org/resources/MSC/2010/>
PREFIX cito: <http://purl.org/spar/cito/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?paper ?title ?year
       (COUNT(DISTINCT ?ref) AS ?refCount)
       (COUNT(DISTINCT ?kw) AS ?kwCount)
       (SAMPLE(?review) AS ?reviewBody)
       (GROUP_CONCAT(DISTINCT STR(?msc); separator=", ") AS ?mscs)
       (GROUP_CONCAT(DISTINCT STR(?kw); separator=", ") AS ?keywords)
WHERE {{
  ?paper dct:issued ?year ;
         dct:subject ?msc ;
         schema:name ?title ;
         schema:keywords ?kw ;
         cito:cites ?ref .

  OPTIONAL {{
    ?paper schema:review ?r .
    ?r schema:reviewBody ?review .
  }}

  FILTER(
    xsd:integer(STR(?year)) >= {year_min} &&
    xsd:integer(STR(?year)) <= {year_max}
  )

  FILTER(
    STRSTARTS(
      STR(?msc),
      "http://msc2010.org/resources/MSC/2010/{msc}"
    )
  )

  FILTER(
    BOUND(?review) && !CONTAINS(LCASE(STR(?review)), "conflicting licenses")
  )
}}
GROUP BY ?paper ?title ?year
HAVING (
  COUNT(DISTINCT ?ref) >= 5 &&
  COUNT(DISTINCT ?kw) >= 3 &&
  COUNT(DISTINCT ?msc) >= 3
)
ORDER BY RAND()
LIMIT {limit}
"""

# ---------------- RUN QUERY PER MSC ----------------
for msc, label in msc_prefixes.items():
    query = query_template.format(
        year_min=YEAR_MIN,
        year_max=YEAR_MAX,
        msc=msc,
        limit=LIMIT_PER_MSC
    )

    sparql.setQuery(query)
    results = sparql.query().convert()

    print(f"\nMSC {msc} — {label} (limit={LIMIT_PER_MSC})")

    for r in results["results"]["bindings"]:
        paper = {
            "paper": r["paper"]["value"],
            "title": r["title"]["value"],
            "year": int(r["year"]["value"]),
            "msc": msc,
            "refs": int(r["refCount"]["value"]),
            "mscs": r["mscs"]["value"].split(", "),
            "keywords": r["keywords"]["value"].split(", "),
            "review": r["reviewBody"]["value"]
        }
        all_papers.append(paper)
        print(f"- {paper['title']} ({paper['year']}) | refs={paper['refs']} | mscs={len(paper['mscs'])} | keywords={len(paper['keywords'])}")

print(f"\nTOTAL PAPERS: {len(all_papers)}")

# ---------------- EXPORT JSONL ----------------
jsonl_path = Path("query-papers-2.jsonl")

with jsonl_path.open("w", encoding="utf-8") as f:
    for row in all_papers:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"JSONL written to: {jsonl_path.resolve()}")
