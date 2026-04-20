import os
import re


def is_docs_noise(file_path: str) -> bool:
    """Shared docs/tutorial detector used by all retrieval-based methods.

    This is intentionally generic: docs/tutorial examples are down-prioritized
    when implementation/source chunks are available, but callers should fall
    back to the original results if filtering would remove everything.
    """
    if "docs_src/" in file_path:
        return True
    if "docs/en/docs/" in file_path:
        return True
    if re.search(r"docs/[a-z]{2}(-[a-z]{2})?/docs/", file_path):
        return True
    if file_path.endswith(".md") and ("docs/" in file_path or file_path.lower() in {"readme.md", "contributing.md"}):
        return True
    return False


def source_first_filter(results: list) -> list:
    """Drop docs/tutorial chunks only when non-doc chunks remain.

    This makes source-first behavior consistent across Method B and Method C
    without silently returning an empty retrieval set for documentation-heavy
    questions.
    """
    source_results = [res for res in results if not is_docs_noise(res["metadata"]["file"])]
    return source_results or results


def query_affinity_bonus(query: str, file_path: str) -> float:
    """
    Conservative affinity bonus. Only triggers on strong matches.
    Returns a multiplier applied to the adjusted distance.
    (Less than 1.0 means lower distance, which is better)
    """
    query_lower = query.lower()
    basename = os.path.basename(file_path).lower()
    stem = basename.rsplit('.', 1)[0]

    # Only exact basename or stem match (e.g., "oauth2.py" in query)
    if basename in query_lower:
        return 0.85  # Mild bonus

    # Small curated map: CamelCase class -> filename
    CLASS_FILE_MAP = {
        "backgroundtasks": "background",
        "uploadfile": "datastructures",
        "httpexception": "exceptions",
        "apirouter": "routing",
        "oauth2passwordbearer": "oauth2",
        "jsonableencoder": "encoders",
    }
    
    # Check for direct map
    for class_name, expected_stem in CLASS_FILE_MAP.items():
        if class_name in query_lower and stem == expected_stem:
            return 0.85

    # Scope contamination logic:
    # If the file belongs to a sample app, but the query mentions a DIFFERENT sample app, penalize heavily.
    sample_apps = ["blank-python", "blank-nodejs", "blank-ruby", "blank-java", "blank-go", "blank-csharp", "blank-rust"]
    if "sample-apps/" in file_path:
        file_app = next((app for app in sample_apps if app in file_path), None)
        query_apps_mentioned = [app for app in sample_apps if app in query_lower or app.replace("-", "") in query_lower]
        if file_app and query_apps_mentioned and file_app not in query_apps_mentioned:
            return 2.5 # Heavy penalty for cross-sample contamination

    # No bonus/penalty otherwise
    return 1.0

def adjust_score(res: dict, query: str = "") -> float:
    """Calculate the reranked distance score using universal penalty table."""
    fp = res['metadata']['file']
    score = res['score']
    
    penalty = 1.0 # Default multiplier
    
    # 1. CORE SOURCE IMPLEMENTATION (Bonus / Priority 0)
    if fp.startswith("fastapi/") and fp.endswith(".py"):
        penalty = 0.75  # Stronger bonus
    elif fp.startswith("sample-apps/blank-python") and fp.endswith(".py"):
        penalty = 0.8  # Strong bonus for specific target app
    elif "sample-apps/" in fp and fp.endswith((".py", ".js", ".go")):
        penalty = 0.8  # Strong bonus for other sample apps
        
    # 2. DOCUMENTATION & TUTORIALS (Penalty / Priority 4)
    elif "docs_src/" in fp:
        penalty = 1.8  # Stronger penalty (usage examples != source impl)
    elif "docs/en/docs/" in fp:
        penalty = 1.7  # English docs
    # Handle translated docs - duplicate content, zero marginal value
    elif re.search(r"docs/[a-z]{2}(-[a-z]{2})?/docs/", fp):
        penalty = 1.9  # Heaviest penalty for i18n duplicates
    elif "ExampleCS/" in fp or fp.endswith(".cs"):
        penalty = 1.5  # Penalize C#
    elif fp.endswith(".md"):
        penalty = 1.2
        
    # 3. CONFIG & IAC FILES
    elif "iam-policies/" in fp and fp.endswith(".json"):
        penalty = 0.85  # IAM policy JSONs ARE the answer
    elif fp.startswith("templates/") and fp.endswith((".yml", ".yaml")):
        penalty = 0.85  # CloudFormation templates ARE the answer
    # Important: root-level templates are also answers in some repos like lambda sample apps
    elif fp.endswith((".yml", ".yaml")) and "template" in fp.lower():
        penalty = 0.85
    elif fp.endswith((".yml", ".yaml")):
        penalty = 1.0   # Don't penalize general yaml
    elif fp.endswith(".json"):
        penalty = 1.3   # Config JSON less useful than code, mild penalty
    elif fp.endswith((".txt", ".ini")):
        penalty = 1.7   # Heavy penalty
        
    final_score = score * penalty
    
    # Apply query affinity
    if query:
        affinity = query_affinity_bonus(query, fp)
        final_score *= affinity
        
    return final_score

def _classify_dataset(file_path: str) -> str:
    """Classify a result into its dataset source."""
    fastapi_root_files = {"pyproject.toml", "README.md", "CONTRIBUTING.md"}
    if (
        file_path.startswith("fastapi/")
        or file_path.startswith("docs_src/")
        or "docs/" in file_path
        or file_path in fastapi_root_files
    ):
        return "fastapi"
    return "lambda"

def rerank_and_select(results: list, top_k: int, query: str = "",
                      min_per_dataset: int = 2) -> list:
    """Sort by adjusted distance, return top_k.
    
    Guarantees at least `min_per_dataset` chunks from each dataset
    if enough candidates exist. This prevents hybrid questions from
    being dominated by a single corpus.
    """
    # Deduplicate results by ID in case they come from multiple queries
    unique_results = []
    seen_ids = set()
    for res in results:
        if res['id'] not in seen_ids:
            seen_ids.add(res['id'])
            unique_results.append(res)
            
    unique_results.sort(key=lambda r: adjust_score(r, query))
    
    if min_per_dataset <= 0 or top_k <= 0:
        return unique_results[:top_k]

    # Dynamic floor for hybrid questions
    q_lower = query.lower()
    is_hybrid = ("fastapi" in q_lower and "lambda" in q_lower)
    if is_hybrid:
        min_per_dataset = max(min_per_dataset, top_k // 2)

    # Split by dataset
    buckets = {}
    for res in unique_results:
        ds = _classify_dataset(res['metadata']['file'])
        buckets.setdefault(ds, []).append(res)
    
    # If only one dataset has results, no floor needed
    if len(buckets) <= 1:
        return unique_results[:top_k]
    
    # Guarantee floor from each dataset
    selected = []
    selected_ids = set()
    for ds, items in buckets.items():
        for item in items[:min_per_dataset]:
            if item['id'] not in selected_ids:
                selected.append(item)
                selected_ids.add(item['id'])
    
    # Fill remaining slots from global ranking
    remaining = top_k - len(selected)
    if remaining > 0:
        for res in unique_results:
            if res['id'] not in selected_ids:
                selected.append(res)
                selected_ids.add(res['id'])
                remaining -= 1
                if remaining <= 0:
                    break
    
    # Re-sort the final selection by adjusted score
    selected.sort(key=lambda r: adjust_score(r, query))
    return selected[:top_k]
