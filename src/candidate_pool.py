import json
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
from string import Template


# ---------------- CONFIG ----------------
endpoint_url = "http://localhost:8890/sparql"
sparql = SPARQLWrapper(endpoint_url)
sparql.setReturnFormat(JSON)

# ---------------- LOAD QUERY PAPERS ----------------
query_jsonl_path = Path("query-papers.jsonl")
query_papers = []

with query_jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        query_papers.append(json.loads(line))

print(f"Loaded {len(query_papers)} query papers from {query_jsonl_path.name}")

# ---------------- RETRIEVAL CORPUS TEMPLATE ----------------

# retrieval_query_template = Template(
retrieval_query_template = Template("""
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX schema: <https://schema.org/>
PREFIX msc: <http://msc2010.org/resources/MSC/2010/>
PREFIX cito: <http://purl.org/spar/cito/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT
  ?paper
  ?title
  ?year
  (SAMPLE(?reviewBody) AS ?reviewBody)
  (GROUP_CONCAT(DISTINCT STR(?kw); separator=" | ") AS ?keywords)
  (GROUP_CONCAT(DISTINCT STR(?msc); separator=" | ") AS ?mscs)
  (GROUP_CONCAT(DISTINCT STR(?ref); separator=" | ") AS ?references)
  (COUNT(DISTINCT ?ref) AS ?refCount)
WHERE {
  ?paper schema:review ?r .
  ?r schema:reviewBody ?reviewBody .

  # Paper metadata
  ?paper dcterms:title ?title ;
         dcterms:issued ?year ;
         dcterms:subject ?msc ;
         cito:cites ?ref .

  OPTIONAL { ?paper schema:keywords ?kw }

  FILTER(xsd:integer(STR(?year)) <= $max_year)
  FILTER(!CONTAINS(LCASE(STR(?reviewBody)), "conflicting licenses"))
}
GROUP BY ?paper ?title ?year
HAVING (
    COUNT(DISTINCT ?ref) > 5 &&
    COUNT(DISTINCT ?msc) >= 3 &&
    COUNT(DISTINCT ?kw) >= 3
)
LIMIT $limit
""")

# ---------------- RUN PER QUERY PAPER ----------------
candidate_pool = {}

for q in query_papers:
    q_year = int(q["year"])
    max_year = q_year - 10  # candidates must be ≥ 10 years older
    limit = 100

    print(f"Processing Query paper {q['paper']} ({q['year']})...")

    query = retrieval_query_template.substitute(max_year=max_year, limit=limit)

    sparql.setQuery(query)
    results = sparql.query().convert()

    candidates = []
    for r in results["results"]["bindings"]:
        candidates.append({
            "paper": r["paper"]["value"],
            "title": r["title"]["value"],
            "year": int(r["year"]["value"]),
            "review": r.get("reviewBody", {}).get("value"),
            "keywords": (
                r["keywords"]["value"].split(" | ")
                if "keywords" in r and r["keywords"]["value"]
                else []
            ),
            "msc_codes": r["mscs"]["value"].split(" | "),
            "references": r["references"]["value"].split(" | "),
            "reference_count": int(r["refCount"]["value"]),
        })

    candidate_pool[q["paper"]] = {
        "query_paper": q,
        "candidates": candidates,
        "candidate_count": len(candidates)
    }

    print(f" → Retrieved {len(candidates)} candidates")

# ---------------- EXPORT JSONL ----------------
jsonl_path = Path("candidate-pool-orig.jsonl")
with jsonl_path.open("w", encoding="utf-8") as f:
    for qid, data in candidate_pool.items():
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Candidate pool JSONL written to: {jsonl_path.resolve()}")
