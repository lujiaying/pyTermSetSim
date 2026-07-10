import os
import urllib.parse
import json
import time

import requests

# Set environment variables
#os.environ['API_KEY'] = 'your_api_key_here'

def do_search(
    cell_name: str,
    api_key: str,
    top_k: int = 3,
) -> dict:
    """
    BioPortal Term Search
    """
    cell_name = urllib.parse.quote(cell_name, safe='')
    url = f"https://data.bioontology.org/search?apikey={api_key}&q={cell_name}&pagesize={top_k}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    else:
        results = response.json()
    return results

def parse_search_results(results: dict) -> str:
    """
    Parses the results from the BioPortal API response.
    We use the top confident mapping for each input name.
    """
    top1 = results['collection'][0]
    mapped_name = top1['prefLabel']
    return mapped_name


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
            for cell_name in input_names:
                results = do_search(cell_name, os.environ['API_KEY'])
                mapped_name = parse_search_results(results)
                overall_results.append(results)
                overall_parsed_results.append((cell_name, mapped_name))
                print(f"Processed cell name: {cell_name}, mapped name: {mapped_name}...")
                time.sleep(0.5)  # sleep to avoid rate limit
            out_items = [{'original_label': original, 'mapped_label': mapped} for original, mapped in overall_parsed_results]
            output = {
                'title': data['title'],
                'rule_version': data['rule_version'],
                'items': out_items,
                'bioportal_raw_response': overall_results,
            }
            out_file_path = os.path.join(out_dir, file_name)
            with open(out_file_path, "w") as f:
                json.dump(output, f, indent=2)


if __name__ == "__main__":
    # Example usage
    """
    cell_name = "FOXP3+ regulatory T (Treg) cells (CD4+ T cells:FOXP3/c2"
    result = do_search(cell_name, os.environ['API_KEY'])
    print(json.dumps(result))
    mapped_name = parse_search_results(result)
    print(f'{cell_name} -> {mapped_name}')
    """

    gen_results_stage2(data_dir="./stage2/", out_dir="./stage2_bioportal_results/")