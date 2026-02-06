import json
import sys
import os
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse
import io


def download_file_from_url(url: str, output_dir: str = "./downloads") -> str:
    """Download a file from CDN URL and save it locally."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        parsed_url = urlparse(url)
        url_parts = parsed_url.path.split('/')
        filename = url_parts[-1] if url_parts else "downloaded_file.json"
        
        if '_$$_' in filename:
            filename = filename.split('_$$_')[-1]
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        output_path = os.path.join(output_dir, filename)
        
        print(f"Downloading file from URL...")
        print(f"  URL: {url}")
        print(f"  Saving to: {output_path}")
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"    File downloaded successfully ({len(response.content)} bytes)")
        
        return output_path
        
    except Exception as e:
        print(f"  ✗ Error downloading file: {e}")
        raise


def check_model_exists(model_name: str, api_base_url: str) -> bool:
    """Check if model is available."""
    try:
        response = requests.get(f"{api_base_url}/api/tags", 
                              headers={'Content-Type': 'application/json'}, 
                              timeout=30)
        response.raise_for_status()
        models = response.json().get('models', [])
        return any(m.get('name') == model_name for m in models)
    except:
        return False


def pull_model(model_name: str, api_base_url: str) -> bool:
    """Pull model if not available."""
    try:
        print(f"  → Pulling model '{model_name}'...")
        response = requests.post(f"{api_base_url}/api/pull",
                               headers={'Content-Type': 'application/json'},
                               json={"name": model_name, "stream": False},
                               timeout=600)
        response.raise_for_status()
        print(f"    Model pulled successfully")
        return True
    except:
        return False


def ensure_model_available(model_name: str, api_base_url: str) -> bool:
    """Ensure model is available."""
    if check_model_exists(model_name, api_base_url):
        print(f"    Model '{model_name}' is available")
        return True
    return pull_model(model_name, api_base_url)


def extract_business_entities(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all business entities and their attributes from the data."""
    entities = {}
    
    business_meaning_requests = data.get("businessMeaningRequest", [])
    
    for bm_request in business_meaning_requests:
        business_entities = bm_request.get("businessEntity", [])
        
        for entity in business_entities:
            be_id = entity.get("beId", "")
            if not be_id:
                continue
            
            attrs = entity.get("attrs", {})
            attributes = []
            for attr_name, attr_type in attrs.items():
                attributes.append({
                    "attribute_name": attr_name,
                    "attribute_type": attr_type,
                    "be_id": be_id
                })
            
            entities[be_id] = {
                "be_id": be_id,
                "name": entity.get("name", ""),
                "description": entity.get("description", ""),
                "category": entity.get("category", ""),
                "linked_br_ids": entity.get("linkedBRIds", []),
                "attributes": attributes
            }
    
    return entities


def get_linked_entities_for_business_process(bp_ids: List[str], data: Dict[str, Any]) -> List[str]:
    """Get all business entity IDs linked to the given business process IDs."""
    linked_be_ids = []
    
    business_meaning_requests = data.get("businessMeaningRequest", [])
    
    for bm_request in business_meaning_requests:
        business_processes = bm_request.get("businessProcess", [])
        
        for bp in business_processes:
            if bp.get("bpId") in bp_ids:
                linked_be_ids.extend(bp.get("linkedBEIds", []))
    
    return list(set(linked_be_ids))


def extract_process_steps_for_bp(data: Dict[str, Any], bp_ids: List[str]) -> List[Dict[str, Any]]:
    """Extract process steps for specific business process IDs."""
    steps = []
    
    business_meaning_requests = data.get("businessMeaningRequest", [])
    
    for bm_request in business_meaning_requests:
        business_processes = bm_request.get("businessProcess", [])
        
        for bp in business_processes:
            if bp.get("bpId") in bp_ids:
                bp_steps = bp.get("steps", [])
                for step in bp_steps:
                    steps.append({
                        "step_id": step.get("stepId"),
                        "step_name": step.get("name"),
                        "description": step.get("description")
                    })
    
    return steps


def extract_prompts(filepath: str) -> tuple:
    """Extract prompts and business entities from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problem_statement = data.get("PROBLEM_STATEMENT", "")
    business_meaning_requests = data.get("businessMeaningRequest", [])
    
    all_entities = extract_business_entities(data)
    
    prompts = []
    for idx, bm_request in enumerate(business_meaning_requests, 1):
        bm = bm_request.get("businessMeaning", {})
        bps = bm_request.get("businessProcess", [])
        
        if not bm:
            continue
        
        linked_bp_ids = bm.get("linkedBPIds", [])
        bp_text = " | ".join([
            f"{bp.get('name', '')}: {bp.get('description', '')}"
            for bp in bps if bp.get("bpId") in linked_bp_ids
        ]) or "No specific business process description available."
        
        linked_be_ids = get_linked_entities_for_business_process(linked_bp_ids, data)
        
        prompt_entities = {}
        all_attributes = []
        
        for be_id in linked_be_ids:
            if be_id in all_entities:
                entity = all_entities[be_id]
                prompt_entities[be_id] = entity
                all_attributes.extend(entity["attributes"])
        
        prompt_text = f"""Task: domain_and_context_extraction

Analyze the following and identify:
1. The broad business domain (e.g., Finance, Healthcare, Manufacturing, Technology)
2. A concise subtopic or specific area within that domain (2-5 words maximum)

Problem Statement:
{problem_statement}

Business Meaning:
{bm.get('description', '')}

Business Process:
{bp_text}

IMPORTANT: The "context" must be a SHORT subtopic or specific area, NOT a full sentence.

Provide response ONLY in this JSON format:
{{
  "domain": "<single broad domain>",
  "context": "<2-5 word subtopic>"
}}"""
        
        prompts.append({
            "prompt_id": f"prompt_{idx}",
            "business_meaning_id": bm.get("bmId", ""),
            "business_meaning_name": bm.get("name", ""),
            "linked_business_process_ids": linked_bp_ids,
            "linked_business_entity_ids": linked_be_ids,
            "business_entities": prompt_entities,
            "all_attributes": all_attributes,
            "prompt": prompt_text
        })
    
    return prompts, all_entities, data


def call_api(prompt_text: str, model_name: str, api_base_url: str) -> str:
    """Call API and return only the response text."""
    try:
        response = requests.post(
            f"{api_base_url}/api/generate",
            headers={'Content-Type': 'application/json'},
            json={
                "model": model_name,
                "prompt": prompt_text,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        return f"Error: {str(e)}"


def process_prompts(prompts: List[Dict[str, Any]], model_name: str, api_base_url: str) -> List[Dict[str, Any]]:
    """Process prompts and get responses."""
    results = []
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\nProcessing {idx}/{len(prompts)}: {prompt['business_meaning_name']}")
        
        response = call_api(prompt['prompt'], model_name, api_base_url)
        
        results.append({
            "prompt_id": prompt['prompt_id'],
            "business_meaning_id": prompt['business_meaning_id'],
            "business_meaning_name": prompt['business_meaning_name'],
            "linked_business_process_ids": prompt['linked_business_process_ids'],
            "linked_business_entity_ids": prompt['linked_business_entity_ids'],
            "business_entities": prompt['business_entities'],
            "all_attributes": prompt['all_attributes'],
            "prompt": prompt['prompt'],
            "response": response.strip()
        })
        
        print(f"    Response received")
    
    return results


def call_llm_for_dependencies(entities: Dict[str, Any], steps: List[Dict[str, Any]], 
                               domain: str, context: str,
                               model_name: str, api_base_url: str) -> str:
    """Call LLM to analyze dependencies and determine inputs/outputs for each entity."""
    
    entity_summary = []
    for be_id, entity in entities.items():
        attrs = [f"{attr['attribute_name']} ({attr['attribute_type']})" 
                for attr in entity['attributes']]
        entity_summary.append(f"{entity['name']} ({be_id}):\n  Attributes: {', '.join(attrs)}")
    
    steps_summary = "\n".join([
        f"Step {step['step_id']}: {step['step_name']}\n  {step['description']}"
        for step in steps
    ])
    
    prompt = f"""Analyze the business entities and process steps to determine data dependencies.

DOMAIN: {domain}
CONTEXT: {context}

BUSINESS ENTITIES:
{chr(10).join(entity_summary)}

PROCESS STEPS (in order):
{steps_summary}

For each business entity, determine:
1. INPUT attributes (data that comes FROM other entities or external sources)
2. OUTPUT attributes (data that is GENERATED by this entity and used by others)

Rules:
- Use the process step order to understand data flow
- Attributes with same/similar names across entities indicate dependencies
- If attribute ends with "_id" and matches another entity name, it's an INPUT from that entity
- Attributes like "created_at", "updated_at" are usually OUTPUTs
- Arrays are usually OUTPUTs

Respond ONLY in this JSON format:
{{
  "entity_dependencies": [
    {{
      "entity_id": "be_xxx",
      "entity_name": "Entity Name",
      "inputs": [
        {{
          "attribute_name": "attr_name",
          "attribute_type": "Type",
          "source_entity_id": "be_yyy or null if external",
          "source_attribute": "source_attr or null"
        }}
      ],
      "outputs": [
        {{
          "attribute_name": "attr_name",
          "attribute_type": "Type"
        }}
      ]
    }}
  ]
}}"""
    
    try:
        print(f"  → Analyzing dependencies")
        response = requests.post(
            f"{api_base_url}/api/generate",
            headers={'Content-Type': 'application/json'},
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        return f"Error: {str(e)}"


def parse_llm_response(llm_response: str) -> Dict[str, Any]:
    """Parse LLM JSON response."""
    try:
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0]
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0]
        
        return json.loads(llm_response.strip())
    except Exception as e:
        print(f"  ⚠ Error parsing LLM response: {e}")
        return {"entity_dependencies": []}


def create_input_output_structures(dependencies: Dict[str, Any], 
                                   include_outputs: bool = True) -> Dict[str, List[Dict]]:
    """Create structured input and input+output data."""
    result = {}
    
    for entity_dep in dependencies.get("entity_dependencies", []):
        entity_id = entity_dep.get("entity_id")
        entity_name = entity_dep.get("entity_name")
        
        attributes = []
        for inp in entity_dep.get("inputs", []):
            attributes.append({
                "name": inp.get("attribute_name"),
                "type": inp.get("attribute_type"),
                "category": "input",
                "source_entity": inp.get("source_entity_id"),
                "source_attribute": inp.get("source_attribute")
            })
        
        if include_outputs:
            for out in entity_dep.get("outputs", []):
                attributes.append({
                    "name": out.get("attribute_name"),
                    "type": out.get("attribute_type"),
                    "category": "output",
                    "source_entity": None,
                    "source_attribute": None
                })
        
        result[entity_id] = {
            "entity_name": entity_name,
            "attributes": attributes
        }
    
    return result


def create_correlation_matrix(structure: Dict[str, Any]) -> Dict[str, Any]:
    """Create correlation matrix showing relationships between attributes."""
    correlations = []
    
    for entity_id, entity_data in structure.items():
        entity_name = entity_data["entity_name"]
        
        for attr in entity_data["attributes"]:
            if attr.get("source_entity") and attr.get("source_attribute"):
                correlations.append({
                    "target_entity": entity_id,
                    "target_entity_name": entity_name,
                    "target_attribute": attr["name"],
                    "target_type": attr["type"],
                    "source_entity": attr["source_entity"],
                    "source_attribute": attr["source_attribute"],
                    "relationship_type": "foreign_key"
                })
    
    unique_correlations = []
    seen = set()
    for corr in correlations:
        key = (corr["target_entity"], corr["target_attribute"], 
               corr["source_entity"], corr["source_attribute"])
        if key not in seen:
            seen.add(key)
            unique_correlations.append(corr)
    
    return {
        "total_correlations": len(unique_correlations),
        "correlations": unique_correlations
    }


def process_dependencies(results_data: Dict[str, Any], input_data: Dict[str, Any],
                        mode: str, model_name: str, api_base_url: str) -> Dict[str, Any]:
    """Process dependencies for each business meaning."""
    all_bm_results = []
    
    for result in results_data["results"]:
        bm_id = result.get("business_meaning_id")
        bm_name = result.get("business_meaning_name")
        
        print(f"\nProcessing dependencies for: {bm_name}")
        print("-"*80)
        
        try:
            response_json = json.loads(result.get("response", "{}"))
            domain = response_json.get("domain", "Unknown")
            context = response_json.get("context", "Unknown")
        except:
            domain = "Unknown"
            context = "Unknown"
        
        print(f"  Domain: {domain}")
        print(f"  Context: {context}")
        
        linked_bp_ids = result.get("linked_business_process_ids", [])
        business_entities = result.get("business_entities", {})
        
        steps = extract_process_steps_for_bp(input_data, linked_bp_ids)
        
        if not business_entities:
            print("  ⚠ No entities, skipping...")
            continue
        
        llm_response = call_llm_for_dependencies(
            business_entities, steps, domain, context, 
            model_name, api_base_url
        )
        
        dependencies = parse_llm_response(llm_response)
        
        input_structure = create_input_output_structures(dependencies, include_outputs=False)
        both_structure = create_input_output_structures(dependencies, include_outputs=True)
        
        input_correlation = create_correlation_matrix(input_structure)
        both_correlation = create_correlation_matrix(both_structure)
        
        print(f"    Dependencies analyzed")
        
        all_bm_results.append({
            "business_meaning_id": bm_id,
            "business_meaning_name": bm_name,
            "domain": domain,
            "context": context,
            "process_steps_count": len(steps),
            "entities_count": len(business_entities),
            "dependencies": dependencies,
            "input_only": {
                "structure": input_structure,
                "correlation_matrix": input_correlation
            },
            "input_and_output": {
                "structure": both_structure,
                "correlation_matrix": both_correlation
            }
        })
    
    return {
        "mode": mode,
        "total_business_meanings": len(all_bm_results),
        "business_meanings": all_bm_results
    }


def map_datatype_to_atsd(attr_type: str) -> Dict[str, Any]:
    """Map attribute type to ATSD distribution configuration."""
    attr_type_lower = attr_type.lower()
    
    if attr_type_lower in ["uuid", "string"]:
        return {
            "type": "categorical",
            "values": ["value1", "value2", "value3"],
            "probabilities": [0.4, 0.3, 0.3]
        }
    elif attr_type_lower in ["int", "integer"]:
        return {
            "type": "uniform",
            "min": 1,
            "max": 1000
        }
    elif attr_type_lower in ["float", "double"]:
        return {
            "type": "float",
            "mean": 50.0,
            "stddev": 10.0
        }
    elif attr_type_lower in ["datetime", "timestamp", "date"]:
        return {
            "type": "date",
            "start_date": "-30d",
            "end_date": "today"
        }
    elif attr_type_lower in ["boolean", "bool"]:
        return {
            "type": "categorical",
            "values": [True, False],
            "probabilities": [0.5, 0.5]
        }
    elif "array" in attr_type_lower or "list" in attr_type_lower:
        return {
            "type": "json",
            "distribution": {
                "type": "ollama_lorax",
                "url": "http://ollama-keda.mobiusdtaas.ai/",
                "model": "gpt-oss:20b",
                "input_type": "string",
                "output_type": "string",
                "context_prompt": "Generate sample data for this attribute"
            }
        }
    else:
        return {
            "type": "categorical",
            "values": ["sample1", "sample2", "sample3"],
            "probabilities": [0.4, 0.3, 0.3]
        }


def create_atsd_payload_for_bm(structure: Dict[str, Any], domain: str, 
                               context: str, row_count: int) -> Dict[str, Any]:
    """Create ATSD payload for ALL entities in a business meaning without edges."""
    nodes = []
    
    # Create nodes for all entities
    for entity_id, entity_data in structure.items():
        entity_name = entity_data["entity_name"]
        attributes = entity_data["attributes"]
        
        if not attributes:
            continue
        
        columns = []
        for attr in attributes:
            attr_name = attr["name"]
            attr_type = attr["type"]
            
            distribution = map_datatype_to_atsd(attr_type)
            
            # If it's string/json type, add context to the prompt
            if attr_type.lower() in ["string", "json"] or "array" in attr_type.lower():
                if distribution.get("type") == "json":
                    # Remove quotes from attribute name and use format for context_prompt
                    clean_attr_name = attr_name.replace("'", "").replace('"', '')
                    clean_domain = domain.replace("'", "").replace('"', '')
                    clean_context = context.replace("'", "").replace('"', '')
                    
                    distribution["distribution"]["context_prompt"] = (
                        f"Generate sample data for attribute {clean_attr_name} "
                        f"in domain {clean_domain} with context {clean_context}. "
                        f"The attribute represents: {clean_attr_name}"
                    )
            
            columns.append({
                "name": attr_name,
                "type": attr_type.lower(),
                "distribution": distribution
            })
        
        nodes.append({
            "id": entity_id,
            "name": entity_name.replace(" ", "_").lower(),
            "row_count": row_count,
            "columns": columns
        })
    
    # Return payload without edges
    return {
        "nodes": nodes,
        "edges": [],
        "constraints": [],
        "response_type": "csv"
    }


def call_atsd_generate(payload: Dict[str, Any], auth_token: str) -> Dict[str, str]:
    """Call ATSD API to generate synthetic data and return the cdn_urls mapping."""
    url = "https://ig.gov-cloud.ai/mobius-synthetic-data-generation/generate-batch"
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"
    }
    
    try:
        print(f"  → Calling ATSD API...")
        print(f"     Nodes: {len(payload['nodes'])}")
        print(f"     Edges: {len(payload['edges'])}")
        
        # Convert payload to JSON string with double quotes
        payload_json = json.dumps(payload, ensure_ascii=False)
        
        # Send request with data parameter (already JSON string)
        response = requests.post(url, headers=headers, data=payload_json, timeout=600)
        response.raise_for_status()
        
        # Parse the response
        response_data = response.json()
        cdn_urls = response_data.get("cdn_urls", {})
        
        print(f"    Data generated - {len(cdn_urls)} entities")
        return cdn_urls
        
    except Exception as e:
        print(f"  ✗ ATSD API error: {e}")
        raise


def map_cdn_urls_to_entities(cdn_urls: Dict[str, str], structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map CDN URLs from ATSD response to business entities."""
    entity_results = []
    
    for entity_id, entity_data in structure.items():
        entity_name = entity_data["entity_name"]
        entity_name_normalized = entity_name.replace(" ", "_").lower()
        
        # Try to find matching URL in cdn_urls
        cdn_url = None
        for key, url in cdn_urls.items():
            # Match by normalized entity name
            if key == entity_name_normalized or entity_id in key:
                cdn_url = url
                break
        
        if cdn_url:
            entity_results.append({
                "entity_id": entity_id,
                "entity_name": entity_name,
                "cdn_url": cdn_url,
                "attribute_count": len(entity_data["attributes"])
            })
            print(f"    Mapped {entity_name} → {cdn_url}")
        else:
            print(f"  ⚠ Warning: No CDN URL found for entity {entity_name}")
    
    return entity_results


def generate_synthetic_data(dependencies_data: Dict[str, Any], row_count: int, 
                           auth_token: str, mode: str) -> Dict[str, Any]:
    """Generate synthetic data for all business entities."""
    
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC DATA")
    print("="*80)
    
    results = []
    
    for bm in dependencies_data["business_meanings"]:
        bm_id = bm["business_meaning_id"]
        bm_name = bm["business_meaning_name"]
        domain = bm["domain"]
        context = bm["context"]
        
        print(f"\nBusiness Meaning: {bm_name}")
        print("-"*80)
        
        # Select structure based on mode
        structure = bm["input_and_output"]["structure"] if mode == "both" else bm["input_only"]["structure"]
        
        if not structure:
            print(f"  Skipping (no entities)")
            continue
        
        print(f"  Total Entities: {len(structure)}")
        
        # Create ONE payload for ALL entities in this BM (without edges)
        payload = create_atsd_payload_for_bm(structure, domain, context, row_count)
        
        if not payload["nodes"]:
            print(f"  Skipping (no valid nodes)")
            continue
        
        # Generate data ONCE for all entities - get CDN URLs directly from ATSD
        cdn_urls = call_atsd_generate(payload, auth_token)
        
        # Map CDN URLs to business entities
        print(f"  → Mapping CDN URLs to entities...")
        bm_results = map_cdn_urls_to_entities(cdn_urls, structure)
        
        results.append({
            "business_meaning_id": bm_id,
            "business_meaning_name": bm_name,
            "entities": bm_results
        })
    
    return {
        "total_business_meanings": len(results),
        "results": results
    }


def main():
    MODEL_NAME = "gpt-oss:20b"
    API_BASE_URL = "http://ollama-keda.mobiusdtaas.ai"
    
    if len(sys.argv) < 5:
        print("Usage: python script.py <cdn_url> <mode> <row_count> <auth_file> [model_name]")
        print("  cdn_url: CDN URL to input JSON")
        print("  mode: 'input' or 'both'")
        print("  row_count: Number of rows to generate")
        print("  auth_file: Path to file containing auth token")
        print("Example: python script.py https://cdn.example.com/file.json both 100 auth.txt")
        sys.exit(1)
    
    cdn_url = sys.argv[1]
    mode = sys.argv[2]
    row_count = int(sys.argv[3])
    auth_file = sys.argv[4]
    if len(sys.argv) >= 6:
        MODEL_NAME = sys.argv[5]
    
    if mode not in ["input", "both"]:
        print("Error: mode must be 'input' or 'both'")
        sys.exit(1)
    
    try:
        # Read auth token
        with open(auth_file, 'r') as f:
            auth_token = f.read().strip()
        
        print("="*80)
        print("SYNTHETIC DATA GENERATION PIPELINE")
        print("="*80)
        print(f"CDN URL: {cdn_url}")
        print(f"Mode: {mode}")
        print(f"Row Count: {row_count}")
        print(f"Model: {MODEL_NAME}")
        print("="*80)
        
        # Download file
        print("\nDownloading file from CDN...")
        input_file = download_file_from_url(cdn_url)
        
        # Check model
        print("\nChecking model...")
        if not ensure_model_available(MODEL_NAME, API_BASE_URL):
            print("✗ Model not available")
            sys.exit(1)
        
        # Step 1: Extract domain/context
        print("\nStep 1: Extracting domain and context...")
        print("="*80)
        prompts, all_entities, input_data = extract_prompts(input_file)
        results = process_prompts(prompts, MODEL_NAME, API_BASE_URL)
        
        results_data = {
            "metadata": {
                "total_prompts": len(prompts),
                "total_business_entities": len(all_entities),
                "model_used": MODEL_NAME
            },
            "all_business_entities": all_entities,
            "results": results
        }
        
        # Step 2: Process dependencies
        print("\nStep 2: Processing dependencies...")
        print("="*80)
        dependencies_data = process_dependencies(
            results_data, input_data, mode, MODEL_NAME, API_BASE_URL
        )
        
        # Step 3: Generate synthetic data
        print("\nStep 3: Generating synthetic data...")
        print("="*80)
        synthetic_data_results = generate_synthetic_data(
            dependencies_data, row_count, auth_token, mode
        )
        
        # Save final output
        output_file = f"synthetic_data_urls_{mode}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(synthetic_data_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*80)
        print(f"  Processing complete")
        print(f"  Results saved to: {output_file}")
        
        # Print summary
        print("\nSummary:")
        print("="*80)
        for bm in synthetic_data_results["results"]:
            print(f"\n{bm['business_meaning_name']}")
            for entity in bm["entities"]:
                print(f"  {entity['entity_name']}: {entity['cdn_url']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()