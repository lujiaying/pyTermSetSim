from typing import List
import requests
import json

def do_map_request(
    input_names: List[str],
) -> dict:
    """
    Maps input names to their corresponding ontology terms.
    Only do web request here.
    """
    url = "https://www.ebi.ac.uk/spot/zooma/v3/api/services/map"
    properties = [{"propertyType": "cell type", "textToMap": name} for name in input_names]
    request_body = {
        "properties": properties,
        "model": "",
        "targetOntologies": ["cl"],
        "includeOtherOntologies": False,
        "filter": {
            "required": ["atlas", "gwas"],
            "preferred": ["atlas"]
        },
        "excludeTermIds": [],
        "returnAll": False,
        "deep": False
    }
    response = requests.post(url, json=request_body)
    results = None
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    return results

def parse_result(results: dict) -> List[str]:
    """
    Parses the results from the ZOOMA API response.
    We use the top confident mapping for each input name.
    """
    parsed_results = []
    for result in results.get("mappings", []):
        original_name = result['textToMap']
        candidates = result['candidates']
        if not candidates:
            # can return null
            mapped_name = None
        else:
            mapped_name = candidates[0]['label']
        parsed_results.append((original_name, mapped_name))
    return parsed_results


if __name__ == "__main__":
    input_names = [
        "T_cells_c6_IFIT1",
        "T_cells_c0_CD4+_CCR7",
        "T_cells_c1_CD4+_IL7R",
        "T_cells_c2_CD4+_T-regs_FOXP3",
        "T_cells_c3_CD4+_Tfh_CXCL13",
        "T_cells_c10_NKT_cells_FCGR3A",
        "T_cells_c9_NK_cells_AREG",
        "T_cells_c5_CD8+_GZMK",
        "T_cells_c11_MKI67",
        "T_cells_c8_CD8+_LAG3",
        "T_cells_c4_CD8+_ZFP36",
        "T_cells_c7_CD8+_IFNG"
    ]
    results = do_map_request(input_names)
    print(json.dumps(results))
    parsed_results = parse_result(results)
    print(json.dumps(parsed_results, indent=2))
