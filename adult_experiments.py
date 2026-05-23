import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn.functional as F

import sys
import atexit

# Import shared modules to eliminate code duplication
from shared_modules import (
    EmbeddingModel, SensitiveAttributeClassifier, PredictionHead,
    GradientReversalLayer, LAFTRAdversary, ROADPerturbation, AdversarialLearningAdversary,
    EmbeddingProjection, Codebook,
    train_phase1, train_phase2, train_group_dro, train_laftr, train_road,
    train_adversarial_learning, train_hsic, train_inlp, train_rlace, compute_hsic_loss,
    evaluate_model, calculate_group_fairness_metrics, calculate_trace_scatter_matrix,
    evaluate_information_leakage, evaluate_mutual_information,
    run_group_inclusion_experiment, print_comparison_summary,
    GroupCodebook, train_phase1_group, train_phase2_group, run_ablation_roar,
    EMBEDDING_DIM, PROJECTION_DIM
)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("experiment_results_adult.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()    

sys.stdout = Logger()
sys.stderr = sys.stdout # Capture errors too
atexit.register(sys.stdout.log.flush)
atexit.register(sys.stdout.log.close)

# Constants now imported from shared_modules
no_sensitive_att_input = False #True
no_target_constraint = False #True
counterfactual_fairness, X_train_s, X_test_s = False, None, None #True
desensitize_test_data = False #True #for EAR (editing augmented regression) method of neutralizing test data
SAE_debias, X_train_MF, X_train_WB = False, None, None #(male, female), (white, black) #for SAE method of neutralizing test data

# 1. Load and Preprocess Data without Scikit-learn (problem w/ installation)
def load_and_preprocess_data():
    """Loads and preprocesses the Adult Census Income dataset using pandas and numpy."""
    global X_train_s, X_test_s, X_train_MF, X_train_WB
    print("Fetching dataset...")
    # adult = fetch_ucirepo(id=2)
    # X = adult.data.features
    # y = adult.data.targets

    #temporary workaround for ucirepo connection issue: until X y assignment lines
    # Use a reliable raw mirror of the Adult dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
    
    # These are the standard UCI columns
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    
    # Load data
    df = pd.read_csv(url, names=columns, na_values='?', skipinitialspace=True)
    
    # Split into X and y to match your original structure
    X = df.drop('income', axis=1)
    y = df[['income']] # Keeping it as a DataFrame so y['income'] works!

    # Drop the 'fnlwgt' column
    X = X.drop('fnlwgt', axis=1)

    # Convert target to binary (0 for <=50K, 1 for >50K)
    y = y['income'].apply(lambda x: 1 if x == '>50K' or x == '>50K.' else 0)
    #y = y.apply(lambda x: 1 if x == '>50K' or x == '>50K.' else 0)

    # Replace '?' with NaN
    X.replace('?', np.nan, inplace=True)

    # Split data first to prevent data leakage
    print("Splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=42)

    #added for CF
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Store the sensitive attributes from the raw training data *before* preprocessing
    X_train_sensitive = X_train_raw[['race', 'sex']].copy()
    X_test_sensitive = X_test_raw[['race', 'sex']].copy()

    if no_sensitive_att_input:
        X_train_raw = X_train_raw.drop('race', axis=1)
        X_train_raw = X_train_raw.drop('sex', axis=1)
        X_test_raw = X_test_raw.drop('race', axis=1)
        X_test_raw = X_test_raw.drop('sex', axis=1)

    # Make copies to avoid SettingWithCopyWarning
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numerical_features = X_train.select_dtypes(include=np.number).columns

    # --- Preprocessing ---
    print("Applying preprocessing...")
    # Imputation
    # Calculate fillers from the training set ONLY
    num_fillers = X_train[numerical_features].median()
    cat_fillers = X_train[categorical_features].mode().iloc[0]

    # Apply to both train and test sets
    X_train[numerical_features] = X_train[numerical_features].fillna(num_fillers)
    X_test[numerical_features] = X_test[numerical_features].fillna(num_fillers)
    X_train[categorical_features] = X_train[categorical_features].fillna(cat_fillers)
    X_test[categorical_features] = X_test[categorical_features].fillna(cat_fillers)

    # X_test_cf = X_test.copy()
    # for i in range(5):
    #     race_value = X_test.loc[i, 'race']
    #     print(f"-- Test sample {i} has race {race_value}, cf has {X_test_cf.loc[i, 'race']}")
    #     if race_value == 'White':
    #         X_test_cf.loc[i, 'race'] = 'Black'
    #     else:
    #         X_test_cf.loc[i, 'race'] = 'White'
    #     print(f"-- Test sample {i} CF race is changed to {X_test_cf.loc[i, 'race']}")

    # Scaling numerical features
    # Calculate scaling factors from the training set ONLY
    mean = X_train[numerical_features].mean()
    std = X_train[numerical_features].std()
    
    X_train[numerical_features] = (X_train[numerical_features] - mean) / std
    X_test[numerical_features] = (X_test[numerical_features] - mean) / std

    if counterfactual_fairness:
        X_train_s, X_test_s = perturb_sensi(X_train), perturb_sensi(X_test)
    
    # One-Hot Encoding for categorical features
    X_train = pd.get_dummies(X_train, columns=categorical_features, dummy_na=False)
    X_test = pd.get_dummies(X_test, columns=categorical_features, dummy_na=False)
    if counterfactual_fairness:
        X_train_s = pd.get_dummies(X_train_s, columns=categorical_features, dummy_na=False)
        X_test_s = pd.get_dummies(X_test_s, columns=categorical_features, dummy_na=False)
    if SAE_debias:
        X_train_MF, X_train_WB = interleave_samples(X_train, 'sex'), interleave_samples(X_train, 'race')
        print("First 4 rows of 'sex_Male' and 'sex_Female' columns:")
        print(X_train_MF[['sex_Male', 'sex_Female']].head(4))
        print("First 4 rows of 'race_White' and 'race_Black' columns:")
        print(X_train_WB[['race_White', 'race_Black']].head(4))
        #X_train_WB = pd.get_dummies(X_train_WB, columns=categorical_features, dummy_na=False)
        #X_train_MF = pd.get_dummies(X_train_MF, columns=categorical_features, dummy_na=False)
    
    # Align columns - crucial for consistent feature sets
    # This ensures X_test has the exact same columns as X_train, filling missing ones with 0
    # and dropping any columns from X_test that were not in X_train.
    X_test = X_test.reindex(columns = X_train.columns, fill_value=0)
    if counterfactual_fairness:
        X_test_s = X_test_s.reindex(columns = X_train_s.columns, fill_value=0)
        X_train_s, X_test_s = X_train_s.values, X_test_s.values
    if SAE_debias:
        X_train_MF, X_train_WB = X_train_MF.values, X_train_WB.values
        print(f"*** X_train_MF sample dim {X_train_MF.shape[1]}, X_train_WB sample dim {X_train_WB.shape[1]}")

    # Return the sensitive attributes for the training set along with the processed data
    return (X_train.values, X_test.values, 
            y_train.values, y_test.values, 
            X_train_sensitive, X_test_sensitive)

def interleave_samples(X_ohe, attribute):
    """
    Creates paired counterfactual samples from an already OHE DataFrame (X_ohe).
    It filters and flips the binary OHE columns directly.
    """
    if attribute == 'sex':
        # Define the binary OHE column names for the attribute
        col1, col2 = 'sex_Male', 'sex_Female'
    elif attribute == 'race':
        # We will focus on two major groups for counterfactuals
        col1, col2 = 'race_White', 'race_Black'
    else:
        raise ValueError(f"Attribute {attribute} not supported for pairing.")

    # --- BLOCK A: Samples where the feature is v1 (e.g., Male or White) ---
    
    # 1. Select original samples where col1 is active (e.g., sex_Male == 1)
    X_v1_original = X_ohe[X_ohe[col1] == 1].copy()
    print(f"Original {col1.split('_')[1]} samples selected: {len(X_v1_original)}")

    # 2. Create counterfactual by FLIPPING the OHE columns
    X_v1_counterfactual = X_v1_original.copy()
    
    # Flip: v1 becomes 0, v2 becomes 1
    X_v1_counterfactual[col1] = 0 # e.g., sex_Male -> 0
    X_v1_counterfactual[col2] = 1 # e.g., sex_Female -> 1

    # Interleave v1 original (even) and v1 counterfactual (odd)
    N1 = len(X_v1_original)
    index_sequence_1 = pd.Series(range(N1))
    X_v1_original = X_v1_original.set_index(index_sequence_1 * 2) 
    X_v1_counterfactual = X_v1_counterfactual.set_index(index_sequence_1 * 2 + 1)
    X_paired1 = pd.concat([X_v1_original, X_v1_counterfactual]).sort_index()

    # --- BLOCK B: Samples where the feature is v2 (e.g., Female or Black) ---

    # 3. Select original samples where col2 is active (e.g., sex_Female == 1)
    X_v2_original = X_ohe[X_ohe[col2] == 1].copy()
    print(f"Original {col2.split('_')[1]} samples selected: {len(X_v2_original)}")

    # 4. Create counterfactual by FLIPPING the OHE columns
    X_v2_counterfactual = X_v2_original.copy()
    
    # Flip: v2 becomes 0, v1 becomes 1
    X_v2_counterfactual[col2] = 0 # e.g., sex_Female -> 0
    X_v2_counterfactual[col1] = 1 # e.g., sex_Male -> 1

    # Interleave v2 counterfactual (even) and v2 original (odd)
    N2 = len(X_v2_original)
    index_sequence_2 = pd.Series(range(N2))
    # NOTE: The desired order for the second block is CF, Original
    X_v2_counterfactual = X_v2_counterfactual.set_index(index_sequence_2 * 2) 
    X_v2_original = X_v2_original.set_index(index_sequence_2 * 2 + 1)
    X_paired2 = pd.concat([X_v2_counterfactual, X_v2_original]).sort_index()
    
    # --- COMBINE AND FINAL CHECK ---
    X_combined = pd.concat([X_paired1, X_paired2], ignore_index=True)
    
    print(f"\nTotal paired samples created: {len(X_combined)}")
    print(f"Final feature count: {X_combined.shape[1]}") # Should be consistent

    return X_combined.reset_index(drop=True)

def perturb_sensi(X):
    X2 = X.copy()
    X2['race'] = X2['race'].replace({
        'White': 'Black',
        'Black': 'White'
    })
    X2['sex'] = X2['sex'].replace({
        'Male': 'Female',
        'Female': 'Male'
    })
    return X2

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    """Manually splits pandas DataFrames into train and test sets."""
    if random_state:
        np.random.seed(random_state)
    
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # Return the raw X data for the sensitive attributes split as well
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

# Model classes, training functions, and evaluation functions now imported from shared_modules

# Dataset-specific functions for Adult dataset (counterfactual fairness evaluation)
def eval_CF_fairness(embedding_model, prediction_head, X_tensor, X_s_tensor, y_tensor):
    embedding_model.eval()
    prediction_head.eval()
    with torch.no_grad():
        embeddings = embedding_model(X_tensor)
        y_pred = prediction_head(embeddings)
        predicted = (y_pred > 0.5).float()
        embeddings_s = embedding_model(X_s_tensor)
        y_pred_s = prediction_head(embeddings_s)
        predicted_s = (y_pred_s > 0.5).float()
        cf_fairness = (predicted == predicted_s).float().mean()
        print(f"Counterfactual fairness is: {cf_fairness.item():.4f}")
        accuracy = (predicted == y_tensor).float().mean()
        print(f"Accuracy of original input: {accuracy.item():.4f}")
        accuracy = (predicted_s == y_tensor).float().mean()
        print(f"Accuracy of counterfactual input: {accuracy.item():.4f}")

# 3. Train and Evaluate the Model
if __name__ == "__main__":
    seed = 9973
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_head = True #False #True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load data
    X_train, X_test, y_train, y_test, X_train_sensitive, X_test_sensitive = load_and_preprocess_data()
    #exit()

    # --- Prepare sensitive attribute data for the training loop ---
    # Define sensitive attributes for attribute factorization
    sensitive_attributes = [['Male', 'Female'], ['White', 'Black']]
    
    # Create the group key strings (e.g., 'Male_White') for backward compatibility
    X_train_sensitive['group_key'] = X_train_sensitive['sex'] + '_' + X_train_sensitive['race']
    
    # Convert group keys to integer labels for the dataset (for backward compatibility)
    group_keys = ['Male_White', 'Male_Black', 'Female_White', 'Female_Black']
    key_to_idx = {key: i for i, key in enumerate(group_keys)}
    
    # Map keys to indices, with -1 for groups we want to ignore
    group_indices = X_train_sensitive['group_key'].map(key_to_idx).fillna(-1).astype(int)
    
    # Create attribute indices for the new factorized codebook
    sex_to_idx = {'Male': 0, 'Female': 1}
    race_to_idx = {'White': 0, 'Black': 1}
    
    # Create attribute indices tensor: shape (batch_size, 2) - first column for sex, second for race
    sex_indices = X_train_sensitive['sex'].map(sex_to_idx).fillna(-1).astype(int)
    race_indices = X_train_sensitive['race'].map(race_to_idx).fillna(-1).astype(int)
    
    # Only keep samples where both attributes are valid
    valid_mask = (sex_indices != -1) & (race_indices != -1)
    
    # Convert data to PyTorch Tensors first
    #X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    try:
        X_train_clean = X_train.astype(np.float32)
        X_train_tensor = torch.tensor(X_train_clean)
        # if counterfactual_fairness:
        #     X_train_s_tensor = torch.tensor(X_train_s.astype(np.float32))
        # if SAE_debias:
        #     X_train_MF_tensor, X_train_WB_tensor = torch.tensor(X_train_MF.astype(np.float32)), torch.tensor(X_train_WB.astype(np.float32))
    except ValueError as e:
        print(f"Error during type conversion: {e}")
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    group_indices_tensor = torch.tensor(group_indices.values, dtype=torch.long)
    
    # Filter tensors to match valid attribute indices
    X_train_tensor_filtered = X_train_tensor[valid_mask]
    y_train_tensor_filtered = y_train_tensor[valid_mask]
    
    # Create attribute indices tensor: shape (batch_size, 2) - first column for sex, second for race
    attribute_indices = torch.stack([
        torch.tensor(sex_indices[valid_mask].values, dtype=torch.long),
        torch.tensor(race_indices[valid_mask].values, dtype=torch.long)
    ], dim=1)
    
    #X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    try:
        X_test_clean = X_test.astype(np.float32)
        X_test_tensor = torch.tensor(X_test_clean)
        # if counterfactual_fairness:
        #     X_test_s_tensor = torch.tensor(X_test_s.astype(np.float32))
    except ValueError as e:
        print(f"Error during type conversion: {e}")
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader with attribute indices
    train_dataset = TensorDataset(X_train_tensor_filtered, y_train_tensor_filtered, attribute_indices)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # Initialize model, loss, and optimizer
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    embedding_model = EmbeddingModel(input_dim)
    prediction_head = PredictionHead()
    embedding_projection = EmbeddingProjection()
    codebook = Codebook(sensitive_attributes=sensitive_attributes, embedding_dim=PROJECTION_DIM)
    criterion = nn.BCELoss()

    # Prepare filtered test data for evaluation
    X_test_sensitive['group_key'] = X_test_sensitive['sex'] + '_' + X_test_sensitive['race']
    test_mask = X_test_sensitive['group_key'].isin(group_keys)
    filtered_X_test_sensitive = X_test_sensitive[test_mask]
    filtered_X_test_tensor = X_test_tensor[test_mask.values]
    true_group_indices = torch.tensor(
        filtered_X_test_sensitive['group_key'].map(key_to_idx).values, 
        dtype=torch.long
    )
    
    # Prepare attribute indices for test data (for compatibility with evaluation functions)
    sex_to_idx = {'Male': 0, 'Female': 1}
    race_to_idx = {'White': 0, 'Black': 1}
    
    test_sex_indices = filtered_X_test_sensitive['sex'].map(sex_to_idx).fillna(-1).astype(int)
    test_race_indices = filtered_X_test_sensitive['race'].map(race_to_idx).fillna(-1).astype(int)
    
    # Only keep samples where both attributes are valid
    test_valid_mask = (test_sex_indices != -1) & (test_race_indices != -1)
    test_attribute_indices = torch.stack([
        torch.tensor(test_sex_indices[test_valid_mask].values, dtype=torch.long),
        torch.tensor(test_race_indices[test_valid_mask].values, dtype=torch.long)
    ], dim=1)

    # Initialize results dictionary for comparison summary
    comparison_results = {}
    
    # --- Phase 1 Training ---
    # combined_params = (
    #     list(embedding_model.parameters()) + 
    #     list(prediction_head.parameters() if train_head else []) +
    #     list(embedding_projection.parameters()) +
    #     list(codebook.parameters())
    # )
    # optimizer_phase1 = optim.Adam(combined_params, lr=0.001)
    
    # train_phase1(embedding_model, prediction_head, embedding_projection, codebook,
    #              train_loader, criterion, optimizer_phase1, n_epochs=25, alpha=2, inter_group_margin=0.5) #inter_group_margin=0.5 alpha=0.5
    for param in embedding_model.parameters(): param.requires_grad = True
    for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
    for param in embedding_projection.parameters(): param.requires_grad = True
    for param in codebook.parameters(): param.requires_grad = True

    param_groups = [
        {"params": embedding_model.parameters(), "lr": 1e-3},
        {"params": prediction_head.parameters(), "lr": 1e-3} if train_head else None,
        {"params": embedding_projection.parameters(), "lr": 1e-3},
        {"params": codebook.parameters(), "lr": 5e-4},
    ]
    # Filter out None values
    param_groups = [pg for pg in param_groups if pg is not None]
    optimizer = optim.Adam(param_groups)
        
    # Phase 1: Metric learning with codebook
    print("\nTraining ROAR Phase 1 ...")
    train_phase1(
        embedding_model, prediction_head, embedding_projection, codebook,
        train_loader, criterion, optimizer, n_epochs=30, alpha=2.0, inter_group_margin=0.5
    ) #alpha=1.0 2.0

    # --- Evaluation after Phase 1 ---
    # Collect metrics after first iteration (ROAR after Phase 1)
    print(f"\n--- Evaluation after Phase 1 (Iteration 1) ---")
    metrics_roar_phase1 = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                                        X_test_tensor, y_test_tensor, X_test_sensitive,
                                        filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                        return_metrics=True)
    comparison_results['ROAR (Phase 1)'] = metrics_roar_phase1

    if no_target_constraint:
        train_head = False

    # --- Phase 2 Training ---
    # optimizer_phase2 = optim.Adam(
    #     list(embedding_model.parameters()) + list(prediction_head.parameters() if train_head else []), 
    #     lr=0.001
    # )
    
    # train_phase2(embedding_model, prediction_head, embedding_projection, codebook,
    #              train_loader, criterion, optimizer_phase2, n_epochs=20, beta=2.0,
    #              use_vae_disentanglement=True, input_dim=input_dim) #beta=0.3
    for param in embedding_model.parameters(): param.requires_grad = True
    for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
    for param in embedding_projection.parameters(): param.requires_grad = False
    for param in codebook.parameters(): param.requires_grad = False

    param_groups_iter_2 = [
        {"params": embedding_model.parameters(), "lr": 1e-3},
        {"params": prediction_head.parameters(), "lr": 1e-4} if train_head else None, #5e-4
    ]
    # Filter out None values
    param_groups_iter_2 = [pg for pg in param_groups_iter_2 if pg is not None]
    optimizer_iter_2 = optim.Adam(param_groups_iter_2)
    train_phase2(
        embedding_model, prediction_head, embedding_projection, codebook,
        train_loader, criterion, optimizer_iter_2, n_epochs=30, beta=2.0, #0.2 beta=2
        use_vae_disentanglement=True, input_dim=input_dim
    )
    print("\n--- Iteration 1 ROAR Evaluation ---")
    metrics_roar_iteration = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                                        X_test_tensor, y_test_tensor, X_test_sensitive,
                                        filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                        return_metrics=True)
    comparison_results['ROAR (iteration 1)'] = metrics_roar_iteration

    # --- Iterative Training ---
    n_iterations = 9 #30 4
    iterative_epochs = 20 # Fewer epochs for each iterative step (originally 5) 5
    patience = 10 #3
    patience_counter = 0
    best_metric_accuracy = float('inf')

    for i in range(n_iterations):
        print(f"\n\n--- ROAR Iteration {i+2}/{n_iterations+1} ---")

        # --- Step 1: Train the adversary (projection and codebook) ---
        print("\nTraining Phase 1 ...")
        # Freeze embedding_model, unfreeze the rest
        # for param in embedding_model.parameters(): param.requires_grad = False
        # for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        # for param in embedding_projection.parameters(): param.requires_grad = True
        # for param in codebook.parameters(): param.requires_grad = True

        # optimizer_iter_1 = optim.Adam(
        #     list(prediction_head.parameters() if train_head else []) +
        #     list(embedding_projection.parameters()) +
        #     list(codebook.parameters()),
        #     lr=0.001
        # )
        
        # train_phase1(embedding_model, prediction_head, embedding_projection, codebook,
        #              train_loader, criterion, optimizer_iter_1, n_epochs=iterative_epochs, 
        #              alpha=2.0, inter_group_margin=0.5, train_embedding_model=False) #inter_group_margin=0.5 alpha=0.5
        for param in embedding_model.parameters(): param.requires_grad = False
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = True
        for param in codebook.parameters(): param.requires_grad = True

        param_groups = [
            {"params": prediction_head.parameters(), "lr": 1e-3} if train_head else None,
            {"params": embedding_projection.parameters(), "lr": 1e-3},
            {"params": codebook.parameters(), "lr": 5e-4},
        ]
        # Filter out None values
        param_groups = [pg for pg in param_groups if pg is not None]
        optimizer = optim.Adam(param_groups)
        
        # Phase 1: Metric learning with codebook
        print("\nTraining ROAR Phase 1 ...")
        train_phase1(
            embedding_model, prediction_head, embedding_projection, codebook,
            train_loader, criterion, optimizer, n_epochs=20, alpha=2.0, inter_group_margin=0.5
        ) #alpha=1.0 2.0

        # Early stopping logic (commented out - metric_accuracy not being set)
        # if metric_accuracy is not None:
        #     if metric_accuracy < best_metric_accuracy:
        #         best_metric_accuracy = metric_accuracy
        #         patience_counter = 0
        #         print(f"  (New best metric accuracy: {best_metric_accuracy:.4f}. Patience counter reset.)")
        #     else:
        #         patience_counter += 1
        #         print(f"  (Metric accuracy did not decrease. Patience counter: {patience_counter}/{patience})")
        #
        #     if patience_counter >= patience:
        #         print(f"\nEarly stopping triggered: Metric accuracy has not decreased for {patience} iterations.")
        #         break

        # --- Step 2: Train the main model to fool the adversary ---
        print("\nTraining ROAR Phase 2...")
        # Unfreeze embedding_model and prediction_head for training
        # for param in embedding_model.parameters(): param.requires_grad = True
        # for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        
        # optimizer_iter_2 = optim.Adam(
        #     list(embedding_model.parameters()) +
        #     list(prediction_head.parameters() if train_head else []),
        #     lr=0.001
        # )
        
        # train_phase2(embedding_model, prediction_head, embedding_projection, codebook,
        #              train_loader, criterion, optimizer_iter_2, n_epochs=iterative_epochs, beta=2.0,
        #              use_vae_disentanglement=True, input_dim=input_dim) #beta=0.3
        for param in embedding_model.parameters(): param.requires_grad = True
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = False
        for param in codebook.parameters(): param.requires_grad = False

        param_groups_iter_2 = [
            {"params": embedding_model.parameters(), "lr": 1e-3},
            {"params": prediction_head.parameters(), "lr": 1e-4} if train_head else None, #5e-4
        ]
        # Filter out None values
        param_groups_iter_2 = [pg for pg in param_groups_iter_2 if pg is not None]
        optimizer_iter_2 = optim.Adam(param_groups_iter_2)
        train_phase2(
            embedding_model, prediction_head, embedding_projection, codebook,
            train_loader, criterion, optimizer_iter_2, n_epochs=20, beta=2.0, #0.2 beta=2
            use_vae_disentanglement=True, input_dim=input_dim
        )
        if i < n_iterations - 1:
            print(f"\n--- Iteration {i+2} ROAR Evaluation ---")
            metrics_roar_iteration = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                                        X_test_tensor, y_test_tensor, X_test_sensitive,
                                        filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                        return_metrics=True)
            comparison_results[f'ROAR (iteration {i+2})'] = metrics_roar_iteration
    
    print("\n--- Final Evaluation ---")
    metrics_roar_final = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                                        X_test_tensor, y_test_tensor, X_test_sensitive,
                                        filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                        return_metrics=True)
    comparison_results['ROAR (Final)'] = metrics_roar_final
    
    # --- ERM Baseline Model Training & Evaluation ---
    print("\n\n--- Training and Evaluating ERM Baseline Model (Empirical Risk Minimization - No Fairness Intervention) ---")
    baseline_embedding_model = EmbeddingModel(input_dim)
    baseline_prediction_head = PredictionHead()
    
    baseline_params = list(baseline_embedding_model.parameters()) + list(baseline_prediction_head.parameters() if train_head else [])
    baseline_optimizer = optim.Adam(baseline_params, lr=0.001)
    baseline_criterion = nn.BCELoss()

    # Simple training loop for the baseline model
    for epoch in range(20): # Using 20 epochs similar to other phases
        baseline_embedding_model.train()
        baseline_prediction_head.train() if train_head else baseline_prediction_head.eval()
        running_loss = 0.0
        # Create a simple dataloader without group info for this training
        baseline_train_dataset = TensorDataset(torch.tensor(X_train.astype(np.float32)), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        baseline_train_loader = DataLoader(dataset=baseline_train_dataset, batch_size=128, shuffle=True)
        
        for inputs, labels in baseline_train_loader:
            baseline_optimizer.zero_grad()
            embeddings = baseline_embedding_model(inputs)
            outputs = baseline_prediction_head(embeddings)
            loss = baseline_criterion(outputs, labels)
            loss.backward()
            baseline_optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/20, ERM Baseline Model Loss: {running_loss/len(baseline_train_loader):.4f}")

    # For baseline evaluation, we don't have a trained codebook or projection, so we pass None
    # We create dummy tensors for the filtered data as they are not used for the main evaluation part
    dummy_filtered_tensor = torch.empty(0, X_test.shape[1])
    dummy_indices_tensor = torch.empty(0, dtype=torch.long)
    dummy_sensitive_df = pd.DataFrame(columns=X_test_sensitive.columns)

    metrics_erm = evaluate_model(baseline_embedding_model, baseline_prediction_head, None, None,
                   torch.tensor(X_test.astype(np.float32)), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1), X_test_sensitive,
                   dummy_filtered_tensor, dummy_indices_tensor, dummy_sensitive_df,
                   return_metrics=True)
    comparison_results['ERM'] = metrics_erm


    # --- Group DRO Baseline Comparison ---
    print("\n\n" + "="*60)
    print("GROUP DRO BASELINE COMPARISON")
    print("="*60)
    print("Starting Group DRO from fresh random initialization for fair comparison...")
    
    # Create fresh randomly initialized models for Group DRO baseline
    # This ensures fair comparison - both methods start from same initialization
    group_dro_embedding_model = EmbeddingModel(input_dim=X_train_tensor.shape[1]).to(device)
    group_dro_prediction_head = PredictionHead().to(device)
    
    # Train Group DRO baseline
    group_dro_optimizer = optim.Adam(
        list(group_dro_embedding_model.parameters()) + 
        list(group_dro_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    group_dro_weights = train_group_dro(
        group_dro_embedding_model, group_dro_prediction_head, train_loader, 
        criterion, group_dro_optimizer, n_epochs=20, eta=0.1, train_embedding_model=True
    )
    
    # Evaluate Group DRO baseline
    print("\n--- Group DRO Baseline Evaluation ---")
    metrics_group_dro = evaluate_model(group_dro_embedding_model, group_dro_prediction_head, None, None,
                                      X_test_tensor, y_test_tensor, X_test_sensitive,
                                      filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                      return_metrics=True)
    comparison_results['Group DRO'] = metrics_group_dro
    
    print("\n--- Comparison Summary ---")
    print(f"Group DRO final weights: {group_dro_weights.cpu().numpy()}")
    print("Group DRO focuses on worst-performing groups by increasing their weights during training.")
    print("This provides a baseline comparison for the attribute-factorized codebook approach.")

    # --- LAFTR Baseline Comparison ---
    print("\n\n" + "="*60)
    print("LAFTR BASELINE COMPARISON")
    print("="*60)
    print("Starting LAFTR from fresh random initialization for fair comparison...")
    
    # Create fresh randomly initialized models for LAFTR baseline
    # This ensures fair comparison - both methods start from same initialization
    laftr_embedding_model = EmbeddingModel(input_dim=X_train_tensor.shape[1]).to(device)
    laftr_task_head = PredictionHead().to(device)
    
    # Create adversary heads for each sensitive attribute
    laftr_adversary_sex = LAFTRAdversary(EMBEDDING_DIM, 2).to(device)  # Male, Female
    laftr_adversary_race = LAFTRAdversary(EMBEDDING_DIM, 2).to(device)  # White, Black
    laftr_adversary_heads = [laftr_adversary_sex, laftr_adversary_race]
    
    # Create adversary criterions (cross-entropy for classification)
    laftr_adv_criterions = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
    
    # Train LAFTR baseline
    laftr_optimizer = optim.Adam(
        list(laftr_embedding_model.parameters()) + 
        list(laftr_task_head.parameters() if train_head else []) +
        list(laftr_adversary_sex.parameters()) +
        list(laftr_adversary_race.parameters()),
        lr=0.001
    )
    
    train_laftr(
        laftr_embedding_model, laftr_task_head, laftr_adversary_heads, train_loader, 
        criterion, laftr_adv_criterions, laftr_optimizer, n_epochs=20, 
        lambda_adv=0.1, lambda_task=1.0, train_embedding_model=True
    )
    
    # Evaluate LAFTR baseline
    print("\n--- LAFTR Baseline Evaluation ---")
    metrics_laftr = evaluate_model(laftr_embedding_model, laftr_task_head, None, None,
                                   X_test_tensor, y_test_tensor, X_test_sensitive,
                                   filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                   return_metrics=True)
    comparison_results['LAFTR'] = metrics_laftr
    
    print("\n--- Comparison Summary ---")
    print("LAFTR uses adversarial training with gradient reversal to learn fair representations.")
    print("The encoder tries to fool adversaries that predict sensitive attributes while maintaining task performance.")
    print("This provides another baseline comparison for the attribute-factorized codebook approach.")

    # --- ROAD Baseline Comparison ---
    print("\n\n" + "="*60)
    print("ROAD BASELINE COMPARISON")
    print("="*60)
    print("Starting ROAD from fresh random initialization for fair comparison...")
    
    # Create fresh randomly initialized models for ROAD baseline
    # This ensures fair comparison - both methods start from same initialization
    road_embedding_model = EmbeddingModel(input_dim=X_train_tensor.shape[1]).to(device)
    road_prediction_head = PredictionHead().to(device)
    road_perturbation_net = ROADPerturbation(X_train_tensor.shape[1], perturbation_dim=32, epsilon=0.1).to(device)
    
    # Train ROAD baseline
    road_optimizer = optim.Adam(
        list(road_embedding_model.parameters()) + 
        list(road_prediction_head.parameters() if train_head else []) +
        list(road_perturbation_net.parameters()),
        lr=0.001
    )
    
    train_road(
        road_embedding_model, road_prediction_head, road_perturbation_net, train_loader,
        criterion, road_optimizer, n_epochs=20, alpha=0.5, train_embedding_model=True
    )
    
    # Evaluate ROAD baseline
    print("\n--- ROAD Baseline Evaluation ---")
    metrics_road = evaluate_model(road_embedding_model, road_prediction_head, None, None,
                                  X_test_tensor, y_test_tensor, X_test_sensitive,
                                  filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                  return_metrics=True)
    comparison_results['ROAD'] = metrics_road
    
    print("\n--- Comparison Summary ---")
    print("ROAD uses robust optimization with adversarial perturbations to handle distribution shifts.")
    print("It minimizes worst-case loss across groups with perturbed inputs to ensure fair performance.")
    print("This provides another baseline comparison for the attribute-factorized codebook approach.")

    # --- AdversarialLearning Baseline Comparison ---
    print("\n\n" + "="*60)
    print("ADVERSARIAL LEARNING BASELINE COMPARISON")
    print("="*60)
    print("Starting AdversarialLearning from fresh random initialization for fair comparison...")
    
    # Create fresh randomly initialized models for AdversarialLearning baseline
    # This ensures fair comparison - both methods start from same initialization
    adv_learning_embedding_model = EmbeddingModel(input_dim=X_train_tensor.shape[1]).to(device)
    adv_learning_task_head = PredictionHead().to(device)
    
    # Single adversary for all sensitive attributes
    adv_learning_adversary = AdversarialLearningAdversary(
        EMBEDDING_DIM, 
        num_attributes=2,  # sex, race
        num_classes_per_attr=[2, 2]  # [Male, Female], [White, Black]
    ).to(device)
    
    # Train AdversarialLearning baseline
    adv_learning_optimizer = optim.Adam(
        list(adv_learning_embedding_model.parameters()) +
        list(adv_learning_task_head.parameters() if train_head else []) +
        list(adv_learning_adversary.parameters()),
        lr=0.001
    )
    
    train_adversarial_learning(
        adv_learning_embedding_model, adv_learning_task_head, adv_learning_adversary,
        train_loader, criterion, nn.CrossEntropyLoss(), adv_learning_optimizer,
        n_epochs=20, alpha=1.0, train_embedding_model=True
    )
    
    # Evaluate AdversarialLearning baseline
    print("\n--- AdversarialLearning Baseline Evaluation ---")
    metrics_adv_learning = evaluate_model(adv_learning_embedding_model, adv_learning_task_head, None, None,
                                          X_test_tensor, y_test_tensor, X_test_sensitive,
                                          filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                          return_metrics=True)
    comparison_results['AdversarialLearning'] = metrics_adv_learning
    
    print("\n--- Comparison Summary ---")
    print("AdversarialLearning uses gradient reversal to remove sensitive information from representations.")
    print("A single adversary predicts all sensitive attributes, and the encoder maximizes adversary loss.")
    print("This is a classic adversarial debiasing method (Zhang et al., 2018) for comparison.")

    # --- HSIC Regularization Baseline Comparison ---
    print("\n\n" + "="*60)
    print("HSIC REGULARIZATION BASELINE COMPARISON")
    print("="*60)
    print("Starting HSIC from fresh random initialization for fair comparison...")
    
    # Create fresh randomly initialized models for HSIC baseline
    # This ensures fair comparison - both methods start from same initialization
    hsic_embedding_model = EmbeddingModel(input_dim=X_train_tensor.shape[1]).to(device)
    hsic_prediction_head = PredictionHead().to(device)
    
    # Train HSIC baseline
    hsic_optimizer = optim.Adam(
        list(hsic_embedding_model.parameters()) + 
        list(hsic_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_hsic(
        hsic_embedding_model, hsic_prediction_head, train_loader,
        criterion, hsic_optimizer, n_epochs=20, lambda_hsic=0.1, train_embedding_model=True
    )
    
    # Evaluate HSIC baseline
    print("\n--- HSIC Baseline Evaluation ---")
    metrics_hsic = evaluate_model(hsic_embedding_model, hsic_prediction_head, None, None,
                                  X_test_tensor, y_test_tensor, X_test_sensitive,
                                  filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                  return_metrics=True)
    comparison_results['HSIC'] = metrics_hsic
    
    # ==================== INLP Baseline ====================
    print("\n--- INLP Baseline ---")
    inlp_embedding_model = EmbeddingModel(input_dim=X_train_tensor.shape[1]).to(device)
    inlp_prediction_head = PredictionHead().to(device)
    inlp_optimizer = optim.Adam(
        list(inlp_embedding_model.parameters()) + 
        list(inlp_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_inlp(
        inlp_embedding_model, inlp_prediction_head, train_loader,
        criterion, inlp_optimizer, n_epochs=20, train_embedding_model=True, num_iterations=10
    )
    
    # Evaluate INLP baseline
    print("\n--- INLP Baseline Evaluation ---")
    metrics_inlp = evaluate_model(inlp_embedding_model, inlp_prediction_head, None, None,
                                  X_test_tensor, y_test_tensor, X_test_sensitive,
                                  filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                  return_metrics=True)
    comparison_results['INLP'] = metrics_inlp
    
    # ==================== RLACE Baseline ====================
    print("\n--- RLACE Baseline ---")
    rlace_embedding_model = EmbeddingModel(input_dim=X_train_tensor.shape[1]).to(device)
    rlace_prediction_head = PredictionHead().to(device)
    rlace_optimizer = optim.Adam(
        list(rlace_embedding_model.parameters()) + 
        list(rlace_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_rlace(
        rlace_embedding_model, rlace_prediction_head, train_loader,
        criterion, rlace_optimizer, n_epochs=20, train_embedding_model=True, num_iterations=10, alpha=100.0
    )
    
    # Evaluate RLACE baseline
    print("\n--- RLACE Baseline Evaluation ---")
    metrics_rlace = evaluate_model(rlace_embedding_model, rlace_prediction_head, None, None,
                                   X_test_tensor, y_test_tensor, X_test_sensitive,
                                   filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                                   return_metrics=True)
    comparison_results['RLACE'] = metrics_rlace
    
    print("\n--- Comparison Summary ---")
    # Print the comparison summary table
    print_comparison_summary(comparison_results)
    
    # ==================== Ablation Study: Group-Level Codebook ====================
    print("\n" + "="*60)
    print("ABLATION STUDY: Group-Level Codebook (No Attribute Factorization)")
    print("="*60)
    
    # Run ablation ROAR with group codebook
    ablation_metrics = run_ablation_roar(
        X_train, y_train, X_train_sensitive,
        X_test, y_test, X_test_sensitive,
        group_keys, attribute_values, device, group_definition,
        train_head=train_head, n_roar_iterations=10, n_epochs_phase1=40, n_epochs_phase2=20,
        alpha=100.0, beta=30.0, inter_group_margin=0.5
    )
    
    comparison_results['Ablation (Group Codebook)'] = ablation_metrics
    
    # Print updated comparison summary with ablation
    print("\n" + "="*60)
    print("COMPARISON SUMMARY (WITH ABLATION)")
    print("="*60)
    print_comparison_summary(comparison_results)
    
    # ==================== Group Inclusion Experiment ====================
    # Configure target group and inclusion percentages
    # Note: group_definition order determines the group_key format (e.g., ['race', 'sex'] -> 'Black_Female')
    seed = 9973
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_group = 'Black_Female'  # Matches group_definition = ['race', 'sex'] 'Black_Female'
    inclusion_percentages = [0.0] #[0.0, 0.1, 0.25, 0.5, 0.75, 0.85]  # 0%, 10%, 25%, 50%, 75%, 85%
    n_roar_iterations = 5  # Number of ROAR iterations
    group_definition = ['race', 'sex']
    
    # Create attribute_values dictionary for the Adult dataset
    attribute_values = {
        'race': ['White', 'Black'],
        'sex': ['Male', 'Female']
    }
    
    results, fairness_metrics = run_group_inclusion_experiment(
        X_train, y_train, X_train_sensitive,
        X_test, y_test, X_test_sensitive,
        target_group, inclusion_percentages,
        attribute_values, device, group_definition, train_head=train_head, n_roar_iterations=n_roar_iterations
    )
    
    print("\n" + "="*60)
    print("ALL BASELINE EXPERIMENTS COMPLETE")
    print("="*60)
    print("\nExperiment setup complete. All baselines have been evaluated.")
