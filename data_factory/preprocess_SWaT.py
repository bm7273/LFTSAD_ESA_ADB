import pandas as pd
import os

def preprocess_swat(input_file, output_file):
    """
    Preprocess SWaT dataset:
    1. Remove Timestamp column
    2. Convert Normal/Attack to 0/1
    """
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Remove Timestamp column
    if ' Timestamp' in df.columns:
        df = df.drop(columns=[' Timestamp'])
        print("✓ Removed Timestamp column")
    
    # 2. Find and convert the label column (Normal/Attack)
    label_col = 'Normal/Attack'
    if label_col in df.columns:
        # Convert: Normal -> 0, Attack -> 1
        df[label_col] = df[label_col].apply(
            lambda x: 0 if str(x).strip() == 'Normal' else 1
        )
        print(f"✓ Converted {label_col} to binary")
        print(f"  Label distribution:\n{df[label_col].value_counts()}")
    
    # 3. Save the processed file
    df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    data_folder = 'dataset/SwaT'
    
    # Process training file
    print("\n" + "="*50)
    print("PROCESSING TRAINING DATA")
    print("="*50)
    preprocess_swat(
        input_file=os.path.join(data_folder, 'swat_train2.csv'),
        output_file=os.path.join(data_folder, 'swat_train2.csv')  # Overwrite
    )
    
    # Process test file
    print("\n" + "="*50)
    print("PROCESSING TEST DATA")
    print("="*50)
    preprocess_swat(
        input_file=os.path.join(data_folder, 'swat2.csv'),
        output_file=os.path.join(data_folder, 'swat2.csv')  # Overwrite
    )
    
    print("\n" + "="*50)
    print("DONE! Both files preprocessed.")
    print("="*50)