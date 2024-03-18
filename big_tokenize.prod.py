import nltk
from nltk.tokenize import word_tokenize
import multiprocessing

# Download the 'punkt' tokenizer if not already downloaded
nltk.download('punkt', quiet=True, raise_on_error=False)

# Path to your input file
input_file_path = "/Users/armenmerikyan/Desktop/wd/python/llm/data/train.csv"

# Path to your output file
output_file_path = "/Users/armenmerikyan/Desktop/wd/python/llm/data/tokenized.txt"

# Define the number of processes (cores) to utilize
num_processes = multiprocessing.cpu_count()

# Function to tokenize lines
def tokenize_lines(lines_chunk):
    tokenized_lines_chunk = []
    for line in lines_chunk:
        tokens = word_tokenize(line)
        tokenized_lines_chunk.append(tokens)
    return tokenized_lines_chunk

if __name__ == '__main__':
    # Read from input file
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    # Split lines into chunks for parallel processing
    chunk_size = len(lines) // num_processes
    line_chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Tokenize lines in parallel
        tokenized_line_chunks = pool.map(tokenize_lines, line_chunks)

    # Flatten the list of tokenized line chunks
    tokenized_lines = [token for chunk in tokenized_line_chunks for token in chunk]

    # Write tokenized lines to output file
    with open(output_file_path, 'w') as output_file:
        for tokens in tokenized_lines:
            output_file.write(' '.join(tokens) + '\n')

    print("Tokenization completed and results written to", output_file_path)
