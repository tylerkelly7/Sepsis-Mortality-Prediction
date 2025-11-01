
"""
src/data_prep.py

Data loading, cleaning, and preprocessing functions for Masters-Thesis project.
Refactored from:
- pymain.ipynb
- Sepsis_Model_Training.ipynb
- main.Rmd
src/data_prep.py

Since the original raw SQL query cannot be replicated, this project uses the
pre-cleaned dataset provided by the authors on GitHub.
"""

# ==================================================
# Part 1
# ==================================================

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

import re
import multiprocessing

from src.utils import resolve_path

# ==================================================
# 1. Load Structured Data
# ==================================================

def load_cleaned_data(path: str) -> pd.DataFrame:
    """
    Load the pre-cleaned structured dataset.

    Args:
        path (str): Path to the cleaned dataset (e.g., Data_after_Cleaning.csv).

    Returns:
        pd.DataFrame: Cleaned structured dataset (one-hot encoded).
    """
    resolved_path = resolve_path(path)
    df = pd.read_csv(resolved_path)
    print(f"✅ Loaded cleaned dataset from {resolved_path} with shape {df.shape}")
    return df


# ==================================================
# 2. Load MIMIC Notes
# ==================================================

def load_mimic_notes(sql_path: str, conn_params: dict, patient_ids: list) -> pd.DataFrame:
    """
    Load clinical notes from MIMIC-IV for specific patients.

    Args:
        sql_path (str): Path to SQL query file (should select notes table).
        conn_params (dict): Database connection parameters.
        patient_ids (list): List of subject_id values to filter.

    Returns:
        pd.DataFrame: Notes for the specified patients.
    """
    # TODO: implement with psycopg2 or sqlalchemy
    # Example pseudocode:
    #
    # conn = psycopg2.connect(**conn_params)
    # with open(sql_path, "r") as f:
    #     query = f.read()
    # # add WHERE subject_id IN (...)
    # ids = ",".join(str(i) for i in patient_ids)
    # query = query.replace("{{PATIENT_IDS}}", ids)
    # df = pd.read_sql(query, conn)
    # conn.close()
    # return df
    raise NotImplementedError("Add SQL logic to filter by patient_ids")

# ==================================================
# 3. Clean Notes
# ==================================================

def clean_notes(df: pd.DataFrame, text_col: str = "note_text") -> pd.DataFrame:
    """
    Apply basic text cleaning steps.

    Args:
        df (pd.DataFrame): DataFrame with notes.
        text_col (str): Column name containing note text.

    Returns:
        pd.DataFrame: Cleaned notes DataFrame.
    """
    # TODO: paste your text cleaning steps from pymain (lowercasing, punctuation removal, etc.)
    return df

# ==================================================
# 4. Truncate Notes
# ==================================================

def truncate_notes(df: pd.DataFrame, text_col: str = "note_text", max_length: int = 500) -> pd.DataFrame:
    """
    Truncate notes to a fixed length for embedding extraction.

    Args:
        df (pd.DataFrame): DataFrame with notes.
        text_col (str): Column name containing note text.
        max_length (int): Maximum number of characters to keep.

    Returns:
        pd.DataFrame: DataFrame with truncated note text.
    """
    df[text_col] = df[text_col].astype(str).str[:max_length]
    return df


# ==================================================
# 5. Train/Test Split (no resampling)
# ==================================================

def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Stratified train/test split for features + target.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of test data (default=0.2).
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y,
                            test_size=test_size, 
                            stratify=y,
                            random_state=random_state
    )
# ==================================================
# Part 2
# ==================================================

import pandas as pd
import re

# ==================================================
# Step 1: Clean Individual Note Text
# ==================================================

def clean_text(text: str) -> str:
    """
    Clean an individual clinical note string.

    Steps:
    - Normalize whitespace
    - Remove underlines
    - Remove unwanted characters but keep clinical symbols
    - Handle missing values

    Args:
        text (str): Raw note text.

    Returns:
        str: Cleaned note text.
    """
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'_+', '', text)    # Remove underlines
    text = re.sub(r'[^\w\s.,:;!?()\-\n]', '', text)  # Remove junk, keep clinical symbols
    return text.strip()

# ==================================================
# Step 2: Process One Group of Notes
# ==================================================

def process_group(record: dict) -> dict:
    """
    Process a grouped record of notes into a single combined string.

    Steps:
    - Extract subject_id, note_type_1, and list of note texts
    - Clean each note using clean_text()
    - Concatenate into one combined note string

    Args:
        record (dict): Dictionary with keys:
            - 'subject_id': patient identifier
            - 'note_type_1': note category (e.g., base vs. addendum radiology)
            - 'text': list of raw note strings

    Returns:
        dict: {
            'subject_id': str/int,
            'note_type_1': str,
            'combined_notes': str (all cleaned notes joined together)
        }
    """
    subject_id = record["subject_id"]
    note_type_1 = record["note_type_1"]
    texts = record["text"]

    cleaned_notes = [clean_text(text) for text in texts]
    combined_notes = " ".join(cleaned_notes)

    return {
        "subject_id": subject_id,
        "note_type_1": note_type_1,
        "combined_notes": combined_notes,
    }

# ==================================================
# Step 3: Group Notes by Subject and Note Type
# ==================================================

def group_notes(df: pd.DataFrame, id_col: str = "subject_id", note_type_col: str = "note_type_1", text_col: str = "text") -> list:
    """
    Group clinical notes by subject and note type, returning records
    suitable for downstream processing.

    Steps:
    - Group by subject_id and note_type_1
    - Aggregate text entries into lists
    - Convert grouped dataframe into list of dict records

    Args:
        df (pd.DataFrame): Input dataframe with subject_id, note_type, and text.
        id_col (str): Column name for patient ID (default: "subject_id").
        note_type_col (str): Column name for note type (default: "note_type_1").
        text_col (str): Column name for note text (default: "text").

    Returns:
        list: List of dictionaries with keys:
            - id_col
            - note_type_col
            - text (list of notes)
    """
    grouped_df = (
        df.groupby([id_col, note_type_col])[text_col]
        .apply(list)
        .reset_index()
    )
    records = grouped_df.to_dict("records")
    return records

from joblib import Parallel, delayed
import multiprocessing

# ==================================================
# Step 4: Parallel Processing of Grouped Notes with Joblib
# ==================================================

def process_notes_in_parallel(records: list, n_jobs: int = None) -> list:
    """
    Process grouped note records in parallel using joblib.

    Args:
        records (list): List of dictionaries, each representing a grouped record
                        (as returned by group_notes()).
        n_jobs (int, optional): Number of CPU cores to use. Defaults to all cores - 1.

    Returns:
        list: List of processed record dictionaries with keys:
            - subject_id
            - note_type_1
            - combined_notes
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1

    processed = Parallel(n_jobs=n_jobs)(
        delayed(process_group)(record) for record in records
    )
    return processed

# ==================================================
# Step 5: Convert Processed Notes to DataFrame and Save
# ==================================================

def save_processed_notes(processed: list, out_path: str) -> pd.DataFrame:
    """
    Convert processed note records into a DataFrame and save to CSV.

    Args:
        processed (list): List of processed note dicts (from process_notes_in_parallel()).
        out_path (str): Path to save the CSV file.

    Returns:
        pd.DataFrame: DataFrame of processed notes, sorted by subject_id and note_type_1.
    """
    df = pd.DataFrame(processed).sort_values(by=["subject_id", "note_type_1"])
    out_path = resolve_path(out_path)
    df.to_csv(out_path, index=False)
    print(f"✅ Processed notes saved to {out_path}")
    return df

# ==================================================
# Step 6: Pivot Notes to Wide Format and Save
# ==================================================

def pivot_notes_to_wide(nlp_long_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Pivot long-format notes DataFrame into wide format with separate
    columns per note type (e.g., Radiology_notes, Discharge_summary_notes).

    Args:
        nlp_long_df (pd.DataFrame): DataFrame with columns:
            - subject_id
            - note_type_1
            - combined_notes
        out_path (str): Path to save the wide-format CSV.

    Returns:
        pd.DataFrame: Wide-format notes DataFrame.
    """
    nlp_wide_df = nlp_long_df.pivot(
        index="subject_id",
        columns="note_type_1",
        values="combined_notes"
    ).reset_index()

    nlp_wide_df.columns.name = None  # Remove category label

    # Rename columns for clarity
    nlp_wide_df = nlp_wide_df.rename(columns={
        "radiology": "Radiology_notes",
        "discharge": "Discharge_summary_notes"
    })

    # Fill missing note values with empty strings
    nlp_wide_df = nlp_wide_df.fillna("")

    # Save to CSV
    out_path = resolve_path(out_path)
    nlp_wide_df.to_csv(out_path, index=False)

    return nlp_wide_df

# ==================================================
# Step 7: Combine Radiology and Discharge Notes per Subject
# ==================================================

def combine_notes(nlp_wide_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Combine radiology and discharge notes into a single column per subject_id.

    Args:
        nlp_wide_df (pd.DataFrame): Wide-format notes DataFrame with columns:
            - subject_id
            - Radiology_notes
            - Discharge_summary_notes
        out_path (str): Path to save the combined CSV.

    Returns:
        pd.DataFrame: DataFrame with subject_id and combined_notes.
    """
    nlp_combined_df = nlp_wide_df.copy()

    nlp_combined_df["combined_notes"] = (
        nlp_combined_df["Radiology_notes"].str.strip() + " " +
        nlp_combined_df["Discharge_summary_notes"].str.strip()
    ).str.strip()

    # Keep only subject_id and combined notes
    nlp_combined_notes_df = nlp_combined_df[["subject_id", "combined_notes"]]

    # Save to CSV
    out_path = resolve_path(out_path)
    nlp_combined_notes_df.to_csv(out_path, index=False)

    return nlp_combined_notes_df

# ==================================================
# Step 8: Merge Notes with Cleaned Structured Dataset
# ==================================================

def merge_notes_with_cleaned(df_clean_path: str,
                             nlp_wide_df: pd.DataFrame,
                             nlp_combined_notes_df: pd.DataFrame,
                             out_path: str) -> pd.DataFrame:
    """
    Merge radiology/discharge notes (wide) and combined notes into the
    pre-cleaned structured dataset.

    Args:
        df_clean_path (str): Path to cleaned structured dataset (Data_after_Cleaning.csv).
        nlp_wide_df (pd.DataFrame): Wide-format notes with Radiology_notes and Discharge_summary_notes.
        nlp_combined_notes_df (pd.DataFrame): Combined notes DataFrame with subject_id + combined_notes.
        out_path (str): Path to save the merged NLP-ready dataset.

    Returns:
        pd.DataFrame: Merged DataFrame with structured data + radiology notes +
                      discharge notes + combined notes.
    """
    # Load cleaned structured dataset
    df_clean = pd.read_csv(df_clean_path)

    # Merge with wide-format notes
    nlp_ready_df = df_clean.merge(
        nlp_wide_df,
        on="subject_id",
        how="left"
    )

    # Merge with combined notes
    nlp_ready_df = nlp_ready_df.merge(
        nlp_combined_notes_df,
        on="subject_id",
        how="left"
    )

    # Save merged dataset
    out_path = resolve_path(out_path)
    nlp_ready_df.to_csv(out_path, index=False)

    return nlp_ready_df

# ==================================================
# Step 9: Inspect DataFrames
# ==================================================

def inspect_dataframes(nlp_wide_df: pd.DataFrame,
                       nlp_combined_notes_df: pd.DataFrame,
                       nlp_ready_df: pd.DataFrame) -> None:
    """
    Print shapes and info for key NLP DataFrames.

    Args:
        nlp_wide_df (pd.DataFrame): Wide-format notes DataFrame.
        nlp_combined_notes_df (pd.DataFrame): Combined notes DataFrame.
        nlp_ready_df (pd.DataFrame): Final merged dataset.

    Returns:
        None
    """
    print("nlp_wide_df shape:", nlp_wide_df.shape)
    print("nlp_combined_notes_df shape:", nlp_combined_notes_df.shape)
    print("nlp_ready_df shape:", nlp_ready_df.shape)
    print("\n=== nlp_ready_df Info ===")
    print(nlp_ready_df.info())

# ==================================================
# Step 10: Write Radiology Notes to Word2Vec Text File
# ==================================================

def write_radiology_notes_for_w2v(nlp_ready_df: pd.DataFrame,
                                  out_path: str = "data/interim/w2v_interim/w2v_Radiology_notes.txt") -> None:
    """
    Write radiology notes to a text file, one note per line, for Word2Vec training.

    Args:
        nlp_ready_df (pd.DataFrame): DataFrame containing 'Radiology_notes'.
        out_path (str): Path to save the Word2Vec training text file.

    Returns:
        None
    """
    out_path = resolve_path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in nlp_ready_df["Radiology_notes"]:
            f.write(str(line).strip() + "\n")


# ==================================================
# Step 11: Write Discharge Notes to Word2Vec Text File
# ==================================================

def write_discharge_notes_for_w2v(nlp_ready_df: pd.DataFrame,
                                  out_path: str = "data/interim/w2v_interim/w2v_Discharge_notes.txt") -> None:
    """
    Write discharge summary notes to a text file, one note per line, for Word2Vec training.

    Args:
        nlp_ready_df (pd.DataFrame): DataFrame containing 'Discharge_summary_notes'.
        out_path (str): Path to save the Word2Vec training text file.

    Returns:
        None
    """
    out_path = resolve_path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in nlp_ready_df["Discharge_summary_notes"]:
            f.write(str(line).strip() + "\n")

# ==================================================
# Step 12: Write Combined Notes to Word2Vec Text File
# ==================================================

def write_combined_notes_for_w2v(nlp_ready_df: pd.DataFrame,
                                 out_path: str = "data/interim/w2v_interim/w2v_combined_notes.txt") -> None:
    """
    Write combined notes (radiology + discharge) to a text file, one note per line,
    for Word2Vec training.

    Args:
        nlp_ready_df (pd.DataFrame): DataFrame containing 'combined_notes'.
        out_path (str): Path to save the Word2Vec training text file.

    Returns:
        None
    """
    out_path = resolve_path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in nlp_ready_df["combined_notes"]:
            f.write(str(line).strip() + "\n")

# =====================================================
# Load Pre-written Note Corpora for Word2Vec
# =====================================================

def load_note_corpus(note_type: str):
    """
    Load pre-tokenized note corpora from data/interim/w2v_interim for Word2Vec.

    Args:
        note_type (str): 'Radiology', 'Discharge', or 'Combined'

    Returns:
        list[list[str]]: Tokenized sentences ready for Word2Vec training.
    """
    from src.utils import resolve_path
    import os

    note_type = note_type.capitalize()
    corpus_map = {
        "Radiology": "data/interim/w2v_interim/w2v_Radiology_notes.txt",
        "Discharge": "data/interim/w2v_interim/w2v_Discharge_notes.txt",
        "Combined":  "data/interim/w2v_interim/w2v_combined_notes.txt",
    }

    if note_type not in corpus_map:
        raise ValueError(f"Unknown note_type '{note_type}'. Expected one of: {list(corpus_map.keys())}")

    corpus_path = resolve_path(corpus_map[note_type])
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f if line.strip()]

    print(f"✅ Loaded {len(sentences):,} sentences for {note_type} corpus.")
    return sentences
