import json
import sys

def extract_tagged_cells(notebook_file, output_file, tags=["#Tag"]):
    print(tags)
    try:
        # Load the notebook
        with open(notebook_file, 'r', encoding='utf-8') as nb_file:
            notebook = json.load(nb_file)

        # Extract tagged cells
        tagged_cells = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source_code = ''.join(cell.get('source', []))
                if any(tag in source_code for tag in tags):
                    tagged_cells.append(source_code)
        
        # Write tagged cells to the output file
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for cell in tagged_cells:
                out_file.write(cell + '\n\n')
        
        print(f"Extracted {len(tagged_cells)} cells with tags '{tags}' to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage: extract_tagged_cells('example_notebook.ipynb', 'output.txt')
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extracter.py <notebook_file.ipynb> <tag1[,tag2]> <output_file.py>")
        print(" e.g., python extracter.py Lab_01.ipynb \"class Agent\" agent.py>")
    else:
        notebook_file = sys.argv[1]
        output_file = sys.argv[3]
        extract_tagged_cells(notebook_file, output_file, tags=sys.argv[2].split(","))

