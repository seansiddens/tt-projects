
import argparse
import sys

def search_ascii(file_path, ascii_string):
    """
    Search for an ASCII string in the binary file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = file.read()
            # Convert the ASCII string to bytes
            search_bytes = ascii_string.encode('utf-8')
            # Search for the bytes in the file
            index = data.find(search_bytes)
            if index != -1:
                print(f"Found ASCII string '{ascii_string}' at offset 0x{index:08x}")
            else:
                print(f"ASCII string '{ascii_string}' not found in the file.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

def search_hex(file_path, hex_string):
    """
    Search for a hexadecimal value in the binary file (little-endian).
    """
    try:
        with open(file_path, 'rb') as file:
            data = file.read()
            # Convert the hex string to bytes (little-endian)
            hex_value = int(hex_string, 16)
            search_bytes = hex_value.to_bytes((hex_value.bit_length() + 7) // 8, byteorder='little')
            # Search for the bytes in the file
            index = data.find(search_bytes)
            if index != -1:
                print(f"Found hex value '{hex_string}' at offset 0x{index:08x}")
            else:
                print(f"Hex value '{hex_string}' not found in the file.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except ValueError:
        print(f"Error: Invalid hex string '{hex_string}'.")
        sys.exit(1)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Search for a message in a binary file.")
    parser.add_argument("file", help="Path to the binary file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ascii", help="Search for an ASCII string")
    group.add_argument("--hex", help="Search for a hexadecimal value (e.g., 0xcafebabe)")
    args = parser.parse_args()

    # Perform the search
    if args.ascii:
        search_ascii(args.file, args.ascii)
    elif args.hex:
        search_hex(args.file, args.hex)

if __name__ == "__main__":
    main()