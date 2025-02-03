import glob
import os
import pandas as pd

# Define the parent directory
parent_dir = "."

# Find all CSV files inside all levels of subdirectories within the parent directory
csv_files = glob.glob(f"{parent_dir}/*/**/*.csv", recursive=True)  # Only searches inside first-level folders

# Create an empty list to store DataFrames
all_dfs = []

for file_path in csv_files:
    file_name = os.path.basename(file_path)  # Extract file name
    folder_name = os.path.basename(os.path.dirname(file_path))  # Extract folder name where the file is stored

    # Extract the first-level directory name (e.g., "9-11", "11-13", "14-16")
    rel_path = os.path.relpath(file_path, parent_dir)  # Get relative path from parent_dir
    first_level_dir = rel_path.split(os.sep)[0]  # Extract first folder in the hierarchy

    # Determine file type based on file name
    if "S1" in file_name:
        file_type = "S1"
    elif "S2" in file_name:
        file_type = "S2"
    elif "S3" in file_name:
        file_type = "S3"
    elif "S4" in file_name:
        file_type = "S4"
    elif "S5" in file_name:
        file_type = "S5"
    elif "S6" in file_name:
        file_type = "S6"
    else:
        file_type = "Unknown"
        print(f"Unknown file type: {file_name}")

    print(f"Processing: {file_name} from {folder_name} - Type: {file_type} - First-Level Dir: {first_level_dir}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Add metadata columns
    df["Date"] = folder_name
    df["StudentID"] = file_type
    #df["TimeSlot"] = first_level_dir  # Add first-level directory to DataFrame
    if first_level_dir == '9-11':
        df["TimePeriod"] = 1
    elif first_level_dir == '11-13':
        df["TimePeriod"] = 2
    elif first_level_dir == '14-16':
        df["TimePeriod"] = 3
    # Append the DataFrame to the list
    all_dfs.append(df)

# Concatenate all DataFrames and save to CSV
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)  # Merge all DataFrames
    final_df.to_csv("processed_motionData2025_1.csv", index=False)  # Save to CSV
    print("All data successfully saved to processed_motionData2025_1.csv")
else:
    print("No CSV files found.")

# import pandas as pd
import os
#
# df1 = pd.read_csv('9-11/02-12-2024/S1-Motion-data 9-11.csv')
# df1["Timeslot"] = 1
#
#
# # Define the folder path
# #folder_path = "9-11/02-12-2024/"
# folder_path = "9-11/02-12-2024/"
# # Loop through all files in the folder
# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".csv"):  # Process only CSV files
#         file_path = os.path.join(folder_path, file_name)  # Get full file path
#
#         # Determine file type (S1 or S2)
#         # file_type = "S1" if "S1" in file_name else "S2" if "S2" in file_name else "Unknown"
#         if "S1" in file_name:
#             file_type = "S1"
#             # print("File belongs to S1")
#         elif "S2" in file_name:
#             file_type = "S2"
#             # print("File belongs to S2")
#         elif "S3" in file_name:
#             file_type = "S3"
#             # print("File belongs to S2")
#         elif "S4" in file_name:
#             file_type = "S4"
#             # print("File belongs to S2")
#         elif "S5" in file_name:
#             file_type = "S5"
#             # print("File belongs to S2")
#         elif "S6" in file_name:
#             file_type = "S6"
#             # print("File belongs to S2")
#         else:
#             print("Unknown file type")
#
#         # Read the CSV
#         df = pd.read_csv(file_path)
#
#         # Add file type as a new column
#         df["StudentID"] = file_type
#
#         # Print first few rows (or save/process further)
#         # print(f"Processed {file_name} - Type: {file_type}")
#         print(df.head())

#
# import glob
# import os
# import pandas as pd
#
# # Define the parent directory
# parent_dir = "."
#
# # Find all CSV files inside all levels of subdirectories
# csv_files = glob.glob(f"{parent_dir}/**/*.csv", recursive=True)
#
# for file_path in csv_files:
#     file_name = os.path.basename(file_path)  # Extract file name
#     folder_name = os.path.basename(os.path.dirname(file_path))  # Extract folder name
#
#     if "S1" in file_name:
#         file_type = "S1"
#     elif "S2" in file_name:
#         file_type = "S2"
#     elif "S3" in file_name:
#         file_type = "S3"
#     elif "S4" in file_name:
#         file_type = "S4"
#     elif "S5" in file_name:
#         file_type = "S5"
#     elif "S6" in file_name:
#         file_type = "S6"
#     else:
#         print("Unknown file type")
#
#     print(f"Processing: {file_name} from {folder_name} - Type: {file_type}")
#
#     df = pd.read_csv(file_path)
#     df["File_Name"] = file_name
#     df["Date"] = folder_name
#     df["StudentID"] = file_type
#
#     print(df.head())  # Show the first few rows of each file
#
# import glob
# import os
# import pandas as pd
#
# # Define the parent directory
# parent_dir = "."
#
# # Find all CSV files inside all levels of subdirectories
# csv_files = glob.glob(f"{parent_dir}/**/*.csv", recursive=True)
#
# for file_path in csv_files:
#     file_name = os.path.basename(file_path)  # Extract file name
#     folder_name = os.path.basename(os.path.dirname(file_path))  # Extract folder name
#
#     # Extract the first-level directory name
#     rel_path = os.path.relpath(file_path, parent_dir)  # Get relative path
#     first_level_dir = rel_path.split(os.sep)[1]  # Get first directory in path
#
#     # Determine file type
#     if "S1" in file_name:
#         file_type = "S1"
#     elif "S2" in file_name:
#         file_type = "S2"
#     elif "S3" in file_name:
#         file_type = "S3"
#     elif "S4" in file_name:
#         file_type = "S4"
#     elif "S5" in file_name:
#         file_type = "S5"
#     elif "S6" in file_name:
#         file_type = "S6"
#     else:
#         file_type = "Unknown"
#         print(f"Unknown file type: {file_name}")
#
#     print(f"Processing: {file_name} from {folder_name} - Type: {file_type} - First-Level Dir: {first_level_dir}")
#
#     df = pd.read_csv(file_path)
#     df["File_Name"] = file_name
#     df["Date"] = folder_name
#     df["StudentID"] = file_type
#     df["First_Level_Dir"] = first_level_dir  # Add first-level directory to DataFrame
#
#     print(df.head())  # Show the first few rows of each file
#
# df.to_csv("processed_motionData2025_1.csv")  # output to csv
#
#
#
# import glob
# import os
# import pandas as pd
#
# # Define the parent directory
# parent_dir = "."
#
# # Find all CSV files inside all levels of subdirectories within the parent directory
# csv_files = glob.glob(f"{parent_dir}/*/**/*.csv", recursive=True)  # Now it only searches inside first-level folders
#
# for file_path in csv_files:
#     file_name = os.path.basename(file_path)  # Extract file name
#     folder_name = os.path.basename(os.path.dirname(file_path))  # Extract folder name where the file is stored
#
#     # Extract the first-level directory name (e.g., "9-11", "11-13", "14-16")
#     rel_path = os.path.relpath(file_path, parent_dir)  # Get relative path from parent_dir
#     first_level_dir = rel_path.split(os.sep)[0]  # Extract first folder in the hierarchy
#
#     # Determine file type based on file name
#     if "S1" in file_name:
#         file_type = "S1"
#     elif "S2" in file_name:
#         file_type = "S2"
#     elif "S3" in file_name:
#         file_type = "S3"
#     elif "S4" in file_name:
#         file_type = "S4"
#     elif "S5" in file_name:
#         file_type = "S5"
#     elif "S6" in file_name:
#         file_type = "S6"
#     else:
#         file_type = "Unknown"
#         print(f"Unknown file type: {file_name}")
#
#     print(f"Processing: {file_name} from {folder_name} - Type: {file_type} - First-Level Dir: {first_level_dir}")
#
#     df = pd.read_csv(file_path)
#     df["File_Name"] = file_name
#     df["Date"] = folder_name
#     df["StudentID"] = file_type
#     df["TimeSlot"] = first_level_dir  # Add first-level directory to DataFrame
#
#     print(df.head())  # Show the first few rows of each file
# df.to_csv("processed_motionData2025_1.csv")  # output to csv