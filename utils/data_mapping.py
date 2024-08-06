import json

def map_data(object_data):
    return {'objects': object_data}

def generate_output(mapped_data, output_path='output_summary.json'):
    with open(output_path, 'w') as f:
        json.dump(mapped_data, f, indent=4)
    print(f"Output saved to {output_path}")
