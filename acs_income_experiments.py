"""
ACS Income Dataset Experiments with Fairness Intervention Methods.
Uses folktables to load ACS Income data and applies the attribute-factorized codebook approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from folktables import ACSDataSource, ACSIncome
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import itertools
import sys
import atexit

# Import shared modules
from shared_modules import (
    EmbeddingModel, PredictionHead, EmbeddingProjection, Codebook,
    GradientReversalLayer, LAFTRAdversary, ROADPerturbation, AdversarialLearningAdversary,
    train_phase1, train_phase2, train_group_dro, train_laftr, train_road,
    train_adversarial_learning, train_hsic, train_inlp, train_rlace, compute_hsic_loss,
    evaluate_model, calculate_group_fairness_metrics, calculate_trace_scatter_matrix,
    evaluate_information_leakage, evaluate_mutual_information,
    EMBEDDING_DIM, PROJECTION_DIM,
    construct_intersectional_groups, create_group_inclusion_split, compute_fairness_metrics,
    run_group_inclusion_experiment, print_comparison_summary,
    GroupCodebook, train_phase1_group, train_phase2_group, run_ablation_roar,
    run_ablation_group_inclusion
)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("experiment_results_acs.log", "a")

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Data Loading ====================

def load_acs_income_data(states=["CA"], survey_year='2018', horizon='1-Year', survey='person'):
    """
    Loads ACS Income data using folktables.
    
    Args:
        states: List of states to include
        survey_year: Year of the survey
        horizon: Time horizon
        survey: Survey type
    
    Returns:
        X: Features DataFrame
        y: Target labels
        sensitive_attributes: Sensitive attributes DataFrame
    """
    print("Loading ACS Income data...")
    
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon=horizon,
        survey=survey
    )
    
    data = data_source.get_data(states=states, download=True)
    
    # Convert to pandas using ACSIncome
    X, y, group = ACSIncome.df_to_pandas(data)
    
    print(f"Loaded {len(X)} samples from {states}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, group

def preprocess_acs_data(X, y, sensitive_attr_cols=['RAC1P', 'SEX', 'SCHL']):
    """
    Preprocesses ACS Income data for fairness experiments.
    
    Args:
        X: Features DataFrame
        y: Target labels
        sensitive_attr_cols: List of sensitive attribute column names
    
    Returns:
        X_processed: Processed features
        y_processed: Processed labels
        sensitive_attributes: Sensitive attributes DataFrame
    """
    print("\nPreprocessing ACS Income data...")
    
    # Store sensitive attributes before preprocessing
    sensitive_attributes = X[sensitive_attr_cols].copy()
    
    # Drop sensitive attributes from features (if needed)
    X_features = X.drop(columns=sensitive_attr_cols)
    
    # Handle categorical variables
    categorical_cols = X_features.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_features[col] = le.fit_transform(X_features[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical variables
    scaler = StandardScaler()
    X_features[numerical_cols] = scaler.fit_transform(X_features[numerical_cols])
    
    # Convert target to binary (income > 50K)
    # y from folktables is already boolean (True/False), so just convert to int
    y_processed = y.astype(int)
    
    print(f"Processed features shape: {X_features.shape}")
    print(f"Target distribution:\n{y_processed.value_counts()}")
    print(f"Sensitive attributes:\n{sensitive_attributes.head()}")
    
    return X_features, y_processed, sensitive_attributes

# ==================== Sensitive Attribute Definition ====================

def bin_schl(schl):
    """Bins SCHL (education level) into 4 categories."""
    if schl <= 15:
        return 0  # Low (less than HS)
    elif schl == 16:
        return 1  # HS graduate
    elif 17 <= schl <= 20:
        return 2  # Some college / associate
    else:
        return 3  # Bachelor's+

def define_sensitive_attributes(sensitive_attributes):
    """
    Defines and processes sensitive attributes for ACS Income.
    
    Sensitive attributes:
    - RAC1P: Race (White, Black, Asian, etc.)
    - SEX: Gender (Male, Female)
    - SCHL: Education level
    
    Returns:
        sensitive_attributes_processed: Processed sensitive attributes
        attribute_mappings: Mapping of original values to indices
    """
    print("\nDefining sensitive attributes...")
    
    # Process race (RAC1P in folktables)
    race_mapping = {
        1: 'White', 2: 'Black', 3: 'American_Indian', 4: 'Alaskan_Native',
        5: 'Asian', 6: 'Pacific_Islander', 7: 'Other', 8: 'Two_or_more'
    }
    sensitive_attributes['RACE'] = sensitive_attributes['RAC1P'].map(race_mapping).fillna('Unknown')
    
    # Process sex
    sex_mapping = {1: 'Male', 2: 'Female'}
    sensitive_attributes['SEX'] = sensitive_attributes['SEX'].map(sex_mapping).fillna('Unknown')
    
    # Process education using binning
    sensitive_attributes['EDUCATION'] = sensitive_attributes['SCHL'].apply(bin_schl).astype(str)
    
    # Select relevant sensitive attributes
    selected_attrs = ['RACE', 'SEX', 'EDUCATION']
    sensitive_attributes_processed = sensitive_attributes[selected_attrs].copy()
    
    print(f"Sensitive attributes shape: {sensitive_attributes_processed.shape}")
    print(f"Unique values per attribute:")
    for attr in selected_attrs:
        print(f"  {attr}: {sensitive_attributes_processed[attr].nunique()} unique values")
    
    return sensitive_attributes_processed

# ==================== Rare/Unseen Group Splits ====================

def design_rare_unseen_splits(sensitive_attributes, attribute_indices, group_keys, 
                               rare_threshold=0.05, unseen_ratio=0.2):
    """
    Designs rare/unseen group splits for evaluating robustness.
    
    Args:
        sensitive_attributes: DataFrame with sensitive attributes
        attribute_indices: Array of attribute indices
        group_keys: List of all group keys
        rare_threshold: Threshold for considering a group as rare (frequency)
        unseen_ratio: Ratio of groups to hold out as unseen
    
    Returns:
        train_mask: Boolean mask for training samples
        test_mask: Boolean mask for test samples
        rare_groups: List of rare groups
        unseen_groups: List of unseen groups
    """
    print("\nDesigning rare/unseen group splits...")
    
    n_samples = len(sensitive_attributes)
    
    # Count actual group occurrences
    actual_group_counts = {}
    for i in range(n_samples):
        group_key = '_'.join(map(str, attribute_indices[i]))
        if group_key not in actual_group_counts:
            actual_group_counts[group_key] = 0
        actual_group_counts[group_key] += 1
    
    # Convert to frequencies
    group_frequencies = {k: v / n_samples for k, v in actual_group_counts.items()}
    
    # Identify rare groups
    rare_groups = [k for k, v in group_frequencies.items() if v < rare_threshold]
    print(f"Number of rare groups (freq < {rare_threshold}): {len(rare_groups)}")
    
    # Identify unseen groups (hold out for testing)
    sorted_groups = sorted(group_frequencies.keys(), key=lambda x: group_frequencies[x])
    n_unseen = int(len(sorted_groups) * unseen_ratio)
    unseen_groups = sorted_groups[:n_unseen]
    print(f"Number of unseen groups (held out): {len(unseen_groups)}")
    
    # Create train/test masks
    train_mask = np.ones(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    
    # Random split with stratification on rare groups
    for i in range(n_samples):
        group_key = '_'.join(map(str, attribute_indices[i]))
        if group_key in unseen_groups:
            # Unseen groups go to test
            train_mask[i] = False
            test_mask[i] = True
        elif np.random.random() < 0.8:  # 80-20 split for other groups
            train_mask[i] = True
            test_mask[i] = False
        else:
            train_mask[i] = False
            test_mask[i] = True
    
    print(f"Training samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")
    
    return train_mask, test_mask, rare_groups, unseen_groups

# ==================== Main Experiment Pipeline ====================

def main():
    """
    Main experiment pipeline for ACS Income dataset.
    """
    print("="*60)
    print("ACS INCOME FAIRNESS INTERVENTION EXPERIMENTS")
    print("="*60)
    
    # 1. Load data
    X, y, group = load_acs_income_data(states=["CA"])
    
    # 2. Preprocess data
    X_processed, y_processed, sensitive_attributes = preprocess_acs_data(X, y)
    
    # 3. Define sensitive attributes
    sensitive_attributes_processed = define_sensitive_attributes(sensitive_attributes)
    
    # Filter out rows with 'Unknown', 'Other', and 'Two_or_more' race to reduce noise
    initial_count = len(sensitive_attributes_processed)
    valid_race_mask = sensitive_attributes_processed['RACE'].isin(['White', 'Black', 'American_Indian', 'Alaskan_Native', 'Asian', 'Pacific_Islander'])
    X_processed = X_processed[valid_race_mask]
    y_processed = y_processed[valid_race_mask]
    sensitive_attributes_processed = sensitive_attributes_processed[valid_race_mask]
    filtered_count = len(sensitive_attributes_processed)
    if initial_count != filtered_count:
        print(f"Filtered out {initial_count - filtered_count} samples with 'Unknown', 'Other', or 'Two_or_more' race")
    
    # 4. Construct intersectional groups
    group_definition = ['RACE', 'SEX', 'EDUCATION']
    group_keys, group_to_idx, attribute_indices, attribute_values, valid_indices = construct_intersectional_groups(
        sensitive_attributes_processed, group_definition
    )
    
    # 4.5 Filter out tiny groups to reduce noise
    min_group_size = 50  # Minimum group size threshold (configurable)
    # Create group_key for each sample
    sensitive_attributes_processed['group_key'] = sensitive_attributes_processed[group_definition].astype(str).agg('_'.join, axis=1)
    group_sizes = sensitive_attributes_processed['group_key'].value_counts()
    valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()
    
    initial_count = len(sensitive_attributes_processed)
    valid_group_mask = sensitive_attributes_processed['group_key'].isin(valid_groups)
    X_processed = X_processed[valid_group_mask]
    y_processed = y_processed[valid_group_mask]
    sensitive_attributes_processed = sensitive_attributes_processed[valid_group_mask]
    attribute_indices = attribute_indices[valid_group_mask]
    filtered_count = len(sensitive_attributes_processed)
    if initial_count != filtered_count:
        print(f"Filtered out {initial_count - filtered_count} samples from groups with size < {min_group_size}")
    
    # 5. Design rare/unseen splits
    train_mask, test_mask, rare_groups, unseen_groups = design_rare_unseen_splits(
        sensitive_attributes_processed, attribute_indices, group_keys
    )
    
    # 6. Split data
    X_train = X_processed[train_mask]
    y_train = y_processed[train_mask]
    X_test = X_processed[test_mask]
    y_test = y_processed[test_mask]
    sensitive_train = sensitive_attributes_processed[train_mask].reset_index(drop=True)
    sensitive_test = sensitive_attributes_processed[test_mask].reset_index(drop=True)
    attribute_indices_train = attribute_indices[train_mask]
    attribute_indices_test = attribute_indices[test_mask]
    
    # 7. Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # 8. Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, torch.tensor(attribute_indices_train))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # 9. Define sensitive attributes for codebook
    sensitive_attr_values = [attribute_values[attr] for attr in group_definition]
    
    # 10. Map column names for evaluation function compatibility
    # Evaluation functions expect lowercase column names (sex, race)
    sensitive_train_mapped = sensitive_train.copy()
    sensitive_test_mapped = sensitive_test.copy()
    
    print("\n" + "="*60)
    print("READY FOR EXPERIMENTS")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input dimension: {X_train.shape[1]}")
    print(f"Number of intersectional groups: {len(group_keys)}")
    print(f"Rare groups: {len(rare_groups)}")
    print(f"Unseen groups: {len(unseen_groups)}")

    input_dim = X_train.shape[1]
    criterion = nn.BCELoss()
    train_head = True  # Set to False if you don't want to train the prediction head
    
    # Initialize comparison results dictionary
    comparison_results = {}
    
    # ==================== ROAR Experiment ====================
    # --- Attribute-Factorized Codebook; Repeated Organization and Removal (ROAR) ---
    print("\n" + "="*60)
    print("ATTRIBUTE-FACTORIZED CODEBOOK (ROAR)")
    print("="*60)
    
    embedding_model = EmbeddingModel(input_dim).to(device)
    prediction_head = PredictionHead().to(device)
    embedding_projection = EmbeddingProjection().to(device)
    
    # Create codebook with sensitive attributes
    codebook = Codebook(sensitive_attr_values, PROJECTION_DIM).to(device)
    
    n_iterations = 10 #30 5
    for t in range(n_iterations):
        print(f"\n\n--- Iteration {t+1}/{n_iterations} ---")
        train_embedding_model = (t == 0) #only train embedding model in first iteration
        #train_embedding_model = True #ablation of freezing emb model in later iterations
        for param in embedding_model.parameters(): param.requires_grad = train_embedding_model
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = True
        for param in codebook.parameters(): param.requires_grad = True

        param_groups = [
            {"params": embedding_model.parameters(), "lr": 1e-4} if train_embedding_model else None,
            {"params": prediction_head.parameters(), "lr": 1e-3} if train_head else None,
            {"params": embedding_projection.parameters(), "lr": 1e-3}, #5e-4
            {"params": codebook.parameters(), "lr": 5e-4}, #1e-4
        ]
        # Filter out None values
        param_groups = [pg for pg in param_groups if pg is not None]
        optimizer = optim.Adam(param_groups)
        
        # Phase 1: Metric learning with codebook
        print("\nTraining ROAR Phase 1 ...")
        train_phase1(
            embedding_model, prediction_head, embedding_projection, codebook,
            #train_loader, criterion, optimizer, n_epochs=20, alpha=2.0, inter_group_margin=0.5,
            train_loader, criterion, optimizer, n_epochs=(40 if t == 0 else 20), alpha=(100.0 if t == 0 else 5.0), inter_group_margin=0.5,
            train_embedding_model=train_embedding_model
        ) #alpha=0.5, n_epochs=20
        # if train_embedding_model:
        #     print("\n--- ROAR Phase 1 Evaluation ---")
        #     metrics_roar_phase1 = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
        #          X_test_tensor, y_test_tensor, sensitive_test_mapped,
        #          X_test_tensor, torch.empty(0), sensitive_test_mapped,
        #          return_metrics=True)
        #     comparison_results['ROAR (Phase 1)'] = metrics_roar_phase1
        
        print("Training ROAR Phase 2 ...")
        for param in embedding_model.parameters(): param.requires_grad = True
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = False #ABLATION True
        for param in codebook.parameters(): param.requires_grad = False #ABLATION True

        param_groups_iter_2 = [
            {"params": embedding_model.parameters(), "lr": 1e-4},
            {"params": prediction_head.parameters(), "lr": 5e-4} if train_head else None,
            #ABLATION ONLY: uncomment next 2 lines
            #{"params": embedding_projection.parameters(), "lr": 1e-3}, #5e-4
            #{"params": codebook.parameters(), "lr": 5e-4}, #1e-4
        ]
        # Filter out None values
        param_groups_iter_2 = [pg for pg in param_groups_iter_2 if pg is not None]
        optimizer_iter_2 = optim.Adam(param_groups_iter_2)
        train_phase2(
            embedding_model, prediction_head, embedding_projection, codebook,
            #train_loader, criterion, optimizer_iter_2, n_epochs=20, beta=2.0, #0.2
            train_loader, criterion, optimizer_iter_2, n_epochs=(40 if t == 0 else 20), beta=(100.0 if t == 0 else 30.0),
            use_vae_disentanglement=True, input_dim=input_dim
        )
        # if t < n_iterations - 1:
        #     print(f"\n--- Iteration {t+1} ROAR Evaluation ---")
        #     metrics_roar_iteration = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
        #                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
        #                 X_test_tensor, torch.empty(0), sensitive_test_mapped,
        #                 return_metrics=True)
        #     comparison_results[f'ROAR (iteration {t+1})'] = metrics_roar_iteration
    
    print("\n--- Final Attribute-Factorized Codebook (ROAR) Evaluation ---")
    metrics_roar_final = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 X_test_tensor, torch.empty(0), sensitive_test_mapped,
                 return_metrics=True)
    comparison_results['ROAR (Final)'] = metrics_roar_final

    '''
    # ==================== Baseline Experiments ====================    
    # --- ERM Baseline (Empirical Risk Minimization) ---
    print("\n" + "="*60)
    print("ERM BASELINE (Empirical Risk Minimization)")
    print("="*60)
    
    erm_embedding_model = EmbeddingModel(input_dim).to(device)
    erm_prediction_head = PredictionHead().to(device)
    
    erm_params = list(erm_embedding_model.parameters()) + list(erm_prediction_head.parameters() if train_head else [])
    erm_optimizer = optim.Adam(erm_params, lr=0.001)
    
    # Simple training loop for ERM baseline
    for epoch in range(20):
        erm_embedding_model.train()
        erm_prediction_head.train() if train_head else erm_prediction_head.eval()
        running_loss = 0.0
        
        for inputs, labels, _ in train_loader:
            erm_optimizer.zero_grad()
            embeddings = erm_embedding_model(inputs)
            outputs = erm_prediction_head(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            erm_optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/20, ERM Loss: {running_loss/len(train_loader):.9f}")
    
    # Evaluate ERM baseline
    print("\n--- ERM Baseline Evaluation ---")
    metrics_erm = evaluate_model(erm_embedding_model, erm_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['ERM'] = metrics_erm
    
    # --- Group DRO Baseline ---
    print("\n" + "="*60)
    print("GROUP DRO BASELINE")
    print("="*60)
    
    group_dro_embedding_model = EmbeddingModel(input_dim).to(device)
    group_dro_prediction_head = PredictionHead().to(device)
    
    group_dro_optimizer = optim.Adam(
        list(group_dro_embedding_model.parameters()) +
        list(group_dro_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_group_dro(
        group_dro_embedding_model, group_dro_prediction_head, train_loader,
        criterion, group_dro_optimizer, n_epochs=20, eta=0.01, train_embedding_model=True
    )
    
    print("\n--- Group DRO Baseline Evaluation ---")
    metrics_group_dro = evaluate_model(group_dro_embedding_model, group_dro_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['Group DRO'] = metrics_group_dro
    
    # --- LAFTR Baseline ---
    print("\n" + "="*60)
    print("LAFTR BASELINE")
    print("="*60)
    
    laftr_embedding_model = EmbeddingModel(input_dim).to(device)
    laftr_task_head = PredictionHead().to(device)
    
    # Create adversaries for each sensitive attribute
    num_sex_classes = len(attribute_values['SEX'])
    num_race_classes = len(attribute_values['RACE'])
    
    laftr_sex_adversary = LAFTRAdversary(EMBEDDING_DIM, num_sex_classes).to(device)
    laftr_race_adversary = LAFTRAdversary(EMBEDDING_DIM, num_race_classes).to(device)
    laftr_adversaries = [laftr_sex_adversary, laftr_race_adversary]
    
    laftr_optimizer = optim.Adam(
        list(laftr_embedding_model.parameters()) +
        list(laftr_task_head.parameters() if train_head else []) +
        list(laftr_sex_adversary.parameters()) +
        list(laftr_race_adversary.parameters()),
        lr=0.001
    )
    
    train_laftr(
        laftr_embedding_model, laftr_task_head, laftr_adversaries, train_loader,
        criterion, [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()], laftr_optimizer, n_epochs=20,
        lambda_adv=0.1, lambda_task=1.0, train_embedding_model=True
    )
    
    print("\n--- LAFTR Baseline Evaluation ---")
    metrics_laftr = evaluate_model(laftr_embedding_model, laftr_task_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['LAFTR'] = metrics_laftr
    
    # --- ROAD Baseline ---
    print("\n" + "="*60)
    print("ROAD BASELINE")
    print("="*60)
    
    road_embedding_model = EmbeddingModel(input_dim).to(device)
    road_prediction_head = PredictionHead().to(device)
    road_perturbation_net = ROADPerturbation(input_dim, perturbation_dim=32, epsilon=0.1).to(device)
    
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
    
    print("\n--- ROAD Baseline Evaluation ---")
    metrics_road = evaluate_model(road_embedding_model, road_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['ROAD'] = metrics_road 

    
    # --- AdversarialLearning Baseline ---
    print("\n" + "="*60)
    print("ADVERSARIAL LEARNING BASELINE")
    print("="*60)
    
    adv_learning_embedding_model = EmbeddingModel(input_dim).to(device)
    adv_learning_task_head = PredictionHead().to(device)
    
    # Single adversary for all sensitive attributes
    num_sex_classes = len(attribute_values['SEX'])
    num_race_classes = len(attribute_values['RACE'])
    adv_learning_adversary = AdversarialLearningAdversary(
        EMBEDDING_DIM,
        num_attributes=2,  # sex, race
        num_classes_per_attr=[num_sex_classes, num_race_classes]
    ).to(device)
    
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
    
    print("\n--- AdversarialLearning Baseline Evaluation ---")
    metrics_adv_learning = evaluate_model(adv_learning_embedding_model, adv_learning_task_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['AdversarialLearning'] = metrics_adv_learning
    
    # --- HSIC Baseline ---
    print("\n" + "="*60)
    print("HSIC REGULARIZATION BASELINE")
    print("="*60)
    
    hsic_embedding_model = EmbeddingModel(input_dim).to(device)
    hsic_prediction_head = PredictionHead().to(device)
    
    hsic_optimizer = optim.Adam(
        list(hsic_embedding_model.parameters()) +
        list(hsic_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_hsic(
        hsic_embedding_model, hsic_prediction_head, train_loader,
        criterion, hsic_optimizer, n_epochs=20, lambda_hsic=0.1, train_embedding_model=True
    )
    
    print("\n--- HSIC Baseline Evaluation ---")
    metrics_hsic = evaluate_model(hsic_embedding_model, hsic_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['HSIC'] = metrics_hsic
    
    # ==================== INLP Baseline ====================
    print("\n--- INLP Baseline ---")
    inlp_embedding_model = EmbeddingModel(input_dim=input_dim).to(device)
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
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['INLP'] = metrics_inlp
    
    # ==================== RLACE Baseline ====================
    print("\n--- RLACE Baseline ---")
    rlace_embedding_model = EmbeddingModel(input_dim=input_dim).to(device)
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
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['RLACE'] = metrics_rlace
    '''
    # Print the comparison summary table
    #from shared_modules import print_comparison_summary
    print_comparison_summary(comparison_results)
    '''
    # ==================== Ablation Study: Group-Level Codebook ====================
    print("\n" + "="*60)
    print("ABLATION STUDY: Group-Level Codebook (No Attribute Factorization)")
    print("="*60)
    
    # Run ablation study with group-level codebook
    ablation_metrics = run_ablation_roar(
        X_train, y_train, sensitive_train,
        X_test, y_test, sensitive_test,
        group_keys, attribute_values, device, group_definition,
        train_head=train_head, n_iterations=10, n_epochs_phase1=40, n_epochs_phase2=20,
        alpha=100.0, beta=30.0, inter_group_margin=0.5
    )
    
    comparison_results['Ablation (Group Codebook)'] = ablation_metrics
    
    # Print updated comparison summary with ablation
    print("\n" + "="*60)
    print("COMPARISON SUMMARY (WITH ABLATION)")
    print("="*60)
    #from shared_modules import print_comparison_summary
    print_comparison_summary(comparison_results)
    
    # ==================== Ablation Group Inclusion Experiment ====================
    # Configure target group for ablation study (0% inclusion)
    # target_group = 'Black_Male_1'
    # n_roar_iterations = 1
    
    # # Run ablation group inclusion experiment with group codebook
    # ablation_inclusion_results = run_ablation_group_inclusion(
    #     X_train, y_train, sensitive_train,
    #     X_test, y_test, sensitive_test,
    #     target_group, group_keys, attribute_values, device, group_definition,
    #     train_head=train_head, n_roar_iterations=n_roar_iterations, n_epochs_phase1=25, n_epochs_phase2=20,
    #     alpha=100.0, beta=30.0, inter_group_margin=0.5
    # )
    
    # ==================== Group Inclusion Experiment ====================
    # Configure target group and inclusion percentages
    target_group = 'Black_Male_1' #Black_Female_1 'Black_Male_1'  # Can be changed to any group key
    inclusion_percentages = [0.0] #[0.0, 0.1, 0.25, 0.5, 0.75, 0.85]  # 0%, 10%, 25%, 50%, 75%, 85% [0.0]
    n_roar_iterations = 5  # Number of ROAR iterations (default: 1) 10
    group_definition = ['RACE', 'SEX', 'EDUCATION']
    
    results, fairness_metrics = run_group_inclusion_experiment(
        X_train, y_train, sensitive_train,
        X_test, y_test, sensitive_test,
        target_group, inclusion_percentages,
        attribute_values, device, group_definition, train_head=train_head, n_roar_iterations=n_roar_iterations
    )
    
    print("\n" + "="*60)
    print("ALL BASELINE EXPERIMENTS COMPLETE")
    print("="*60)
    print("\nExperiment setup complete. All baselines have been evaluated.")
    '''
if __name__ == "__main__":
    main()
