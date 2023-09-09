import argparse
import torch

def convert_pt_to_bin(input_pt_file, output_bin_file):
    tensor = torch.load(input_pt_file, map_location=torch.device('cpu'))
    numpy_array = tensor.cpu().numpy()

    numpy_array.tofile(output_bin_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .pt to .bin')
    parser.add_argument('--input', type=str, required=True, help='Input .pt file')
    parser.add_argument('--output', type=str, required=True, help='Output .bin file')

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    convert_pt_to_bin(input_file, output_file)
