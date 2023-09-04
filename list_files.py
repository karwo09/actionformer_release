import os

# Specify the directory path you want to list files from
directory_path = "/home/karolwojtulewicz/code/actionformer_release/data/anet_1.3/tsp_features"

# Ensure the directory exists
if not os.path.exists(directory_path):
    print(f"The directory '{directory_path}' does not exist.")
else:
    # Get a list of all files in the directory
    file_names = [f.replace("v_","").replace(".npy","") for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Specify the name of the output text file where the names will be saved
    output_file = "file_names.txt"

    # Write the file names to the output text file
    with open(output_file, "w") as file:
        file.write("\n".join(file_names))

    print(f"File names saved to '{output_file}'.")
