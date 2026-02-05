import json
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

# ---------------- CONFIG ----------------
ENDPOINT_URL = "http://localhost:8890/sparql"
INPUT_JSONL = Path("candidate-pool-no-cited.jsonl")
OUTPUT_JSONL = Path("candidate-pool.jsonl")

MAX_QUERY_REFS = 20

sparql = SPARQLWrapper(ENDPOINT_URL)
sparql.setReturnFormat(JSON)

# ---------------- SPARQL ----------------

# 1) Get references of the query paper
SPARQL_GET_REFS = """
PREFIX cito: <http://purl.org/spar/cito/>

SELECT DISTINCT ?ref
WHERE {
  <QUERY_PAPER> cito:cites ?ref .
}
"""

# 2) Get metadata for reference papers
SPARQL_GET_METADATA = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX schema: <https://schema.org/>
PREFIX cito: <http://purl.org/spar/cito/>

SELECT
  ?paper
  ?title
  ?year
  (SAMPLE(?reviewBody) AS ?reviewBody)
  (GROUP_CONCAT(DISTINCT STR(?kw); separator=" | ") AS ?keywords)
  (GROUP_CONCAT(DISTINCT STR(?msc); separator=" | ") AS ?mscs)
  (GROUP_CONCAT(DISTINCT STR(?r2); separator=" | ") AS ?references)
  (COUNT(DISTINCT ?r2) AS ?refCount)
WHERE {
  VALUES ?paper { %s }

  ?paper dcterms:title ?title ;
         dcterms:issued ?year .

  OPTIONAL {
    ?paper schema:review ?rv .
    ?rv schema:reviewBody ?reviewBody .
  }

  OPTIONAL { ?paper schema:keywords ?kw }
  OPTIONAL { ?paper dcterms:subject ?msc }
  OPTIONAL { ?paper cito:cites ?r2 }
}
GROUP BY ?paper ?title ?year
"""

# ---------------- HELPERS ----------------
def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, data):
    with path.open("w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def get_query_references(query_uri):
    sparql.setQuery(SPARQL_GET_REFS.replace("<QUERY_PAPER>", f"<{query_uri}>"))
    bindings = sparql.query().convert()["results"]["bindings"]
    return [b["ref"]["value"] for b in bindings]

def get_metadata(paper_uris):
    if not paper_uris:
        return {}

    values = " ".join(f"<{u}>" for u in paper_uris)
    sparql.setQuery(SPARQL_GET_METADATA % values)
    bindings = sparql.query().convert()["results"]["bindings"]

    meta = {}
    for row in bindings:
        uri = row["paper"]["value"]
        meta[uri] = {
            "paper": uri,
            "title": row.get("title", {}).get("value", ""),
            "year": int(row.get("year", {}).get("value", "0")),
            "review": row.get("reviewBody", {}).get("value", ""),
            "keywords": row.get("keywords", {}).get("value", "").split(" | ") if row.get("keywords") else [],
            "msc_codes": row.get("mscs", {}).get("value", "").split(" | ") if row.get("mscs") else [],
            "references": row.get("references", {}).get("value", "").split(" | ") if row.get("references") else [],
            "reference_count": int(row.get("refCount", {}).get("value", "0")),
            "_relevance": 0.0, 
            "llm_score": 0.0,
            "is_cited": 1
        }
    return meta

# ---------------- MAIN ----------------

def main():
    records = load_jsonl(INPUT_JSONL)

    for i, record in enumerate(records):
        print (f"# Processing {i}...")
        query_uri = record["query_paper"]["paper"]

        print (f"Getting references...")
        refs = get_query_references(query_uri)
        record["query_paper"]["references"] = refs

        # --- Step 2: enrich candidates with these refs ---
        refs = refs[:MAX_QUERY_REFS]
        print (f"Getting metadata...")
        metadata = get_metadata(refs)

        existing = {c["paper"] for c in record.get("candidates", [])}

        for ref_uri in refs:
            if ref_uri in metadata and ref_uri not in existing:
                record.setdefault("candidates", []).append(metadata[ref_uri])

    save_jsonl(OUTPUT_JSONL, records)
    print(f"Saved enriched dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
