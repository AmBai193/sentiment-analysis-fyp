import pandas as pd
from deep_translator import GoogleTranslator
import os
import time


def translate_excel():
    # Get filename from user
    file_path = input("Enter the filename (must be in the same directory): ")

    # Check if file exists
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    # Determine file format
    file_extension = os.path.splitext(file_path)[-1].lower()
    try:
        if file_extension == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            print("Error: Unsupported file format. Please use CSV or Excel (XLSX/XLS).")
            return
    except UnicodeDecodeError:
        print("UTF-8 decoding failed. Trying ISO-8859-1 encoding...")
        df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Check if 'Comment' column exists
    if 'Comment' not in df.columns:
        print("Error: 'Comment' column not found in the file.")
        return

    translator = GoogleTranslator(source='ms', target='en')
    translated_comments = []
    total_rows = len(df)
    failed_count = 0  # Counter for failed translations

    # Translate in chunks
    for index, comment in enumerate(df['Comment'].fillna("")):
        if comment.strip() == "":
            translated_comments.append("")  # Keep empty comments empty
        else:
            try:
                translated_text = translator.translate(comment)
                translated_comments.append(translated_text)
            except Exception:
                translated_comments.append("[Translation Failed]")
                failed_count += 1

        # Display progress every 100 lines
        if (index + 1) % 100 == 0 or index + 1 == total_rows:
            print(f"Progress: {index + 1}/{total_rows} comments translated...")

        time.sleep(0.5)  # Prevent hitting API rate limits (adjust as needed)

    # Add new column to DataFrame
    df.insert(df.columns.get_loc('Comment') + 1, 'Translated_Comment', translated_comments)

    # Save back to file
    output_path = "translated_" + file_path
    if file_extension == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_path, index=False, engine="openpyxl")

    print(f"✅ Translation completed! File saved as {output_path}")
    print(f"❌ Total rows with translation failed: {failed_count}")


# Run the function
translate_excel()
