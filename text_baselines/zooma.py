from typing import List, Iterator, Tuple, TypeVar
import os
import json
import time

import requests

T = TypeVar('T')
def batched(lst: List[T], n: int) -> Iterator[Tuple[T, ...]]:
    """quick implementation from py3.12 batched"""
    for i in range(0, len(lst), n):
        yield tuple(lst[i:i + n])


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
        #"targetOntologies": ["cl"],
        "targetOntologies": ["cl", "clo", "hcao", "pcl"],
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


def gen_results_stage2(
    data_dir: str,
    out_dir: str,
):
    """
    Use pipeline stage 2 data as input
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # read json files from data_dir
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            # ad-hoc skip one json file
            if file_name == "paper_label_data_annotation_mapping.summary.json":
                continue
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")
            with open(file_path, "r") as f:
                data = json.load(f)
            input_names = [item['original_label'] for item in data['items']]
            overall_results = []
            overall_parsed_results = []
            for batch_names in batched(input_names, 10):
                results = do_map_request(batch_names)
                parsed_results = parse_result(results)
                overall_results.extend(results['mappings'])
                overall_parsed_results.extend(parsed_results)
                print(f"Processed batch: {batch_names}, parsed results: {parsed_results}...")
                time.sleep(1)  # sleep for 1 second to avoid rate limit
            out_items = [{'original_label': original, 'mapped_label': mapped} for original, mapped in overall_parsed_results]
            output = {
                'title': data['title'],
                'rule_version': data['rule_version'],
                'items': out_items,
                'zooma_raw_response': overall_results,
            }
            out_file_path = os.path.join(out_dir, file_name)
            with open(out_file_path, "w") as f:
                json.dump(output, f, indent=2)


if __name__ == "__main__":
    # test case 1
    """
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
    """

    #gen_results_stage2(data_dir="./stage2/", out_dir="./stage2_zooma_4onto_results/")
    #gen_results_stage2(data_dir="./all_journals_gpt55_thinking_20260711/stage2/", out_dir="./all_journals_gpt55_thinking_20260711_stage2_zooma_4onto_results/")
    gen_results_stage2(data_dir="./all_journals_deepseek_v4_pro_thinking_20260711/stage2/", out_dir="./all_journals_deepseek_v4_pro_thinking_20260711_stage2_zooma_4onto_results/")