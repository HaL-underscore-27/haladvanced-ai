import csv

def convert_shorthand(value):
    """Convert shorthand notations like 1.2M, 1.1k, 846.26K, or -19.12% to full numbers."""
    if isinstance(value, str):
        if value.endswith('M'):
            return int(float(value[:-1]) * 1_000_000)
        elif value.endswith('k') or value.endswith('K'):
            return int(float(value[:-1]) * 1_000)
        elif value.endswith('%'):
            return float(value[:-1]) / 100  # Convert percentage to decimal
        try:
            return float(value)  # Convert numeric strings to float
        except ValueError:
            pass  # Leave non-numeric strings as is
    return value

def process_csv(input_file, output_file):
    """Read a CSV file, process shorthand notations, and write to a new file."""
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader]

    # Process the rows
    processed_rows = []
    for row in rows:
        processed_row = [convert_shorthand(cell) for cell in row]
        processed_rows.append(processed_row)

    # Write the processed rows to a new file
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

# File paths
input_csv = 'files/solana.csv'
output_csv = 'files/solana_processed.csv'

# Process the file
process_csv(input_csv, output_csv)