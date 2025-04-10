import torch
from config import N_EPOCHS, DEVICE, MODELS, LABELS_CODES, TARGET_MODE, H_DIM, USE_FINGERPRINT, N_RUNS
from alternative_dataset_builder import train_dataloader, val_dataloader

GIN_TRAIN_PREC_LIST = []
GIN_TRAIN_REC_LIST = []
GIN_TRAIN_F1_LIST = []
GIN_VAL_PREC_LIST = []
GIN_VAL_REC_LIST = []
GIN_VAL_F1_LIST = []

GINE_TRAIN_PREC_LIST = []
GINE_TRAIN_REC_LIST = []
GINE_TRAIN_F1_LIST = []
GINE_VAL_PREC_LIST = []
GINE_VAL_REC_LIST = []
GINE_VAL_F1_LIST = []

MPL_TRAIN_PREC_LIST = []
MPL_TRAIN_REC_LIST = []
MPL_TRAIN_F1_LIST = []
MPL_VAL_PREC_LIST = []
MPL_VAL_REC_LIST = []
MPL_VAL_F1_LIST = []

for MODEL in MODELS:
    print(f"Training {MODEL.upper()} model")
    
    if MODEL == "gin":
        N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
        from models.GIN import GIN, GINWithEdgeFeatures
        model = GIN(num_node_features=N_FEATURES, 
                    dim_h=H_DIM, 
                    num_classes=len(LABELS_CODES.keys())).to(DEVICE) #, num_heads=4
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        loss_criterion = torch.nn.CrossEntropyLoss() if "hot" in TARGET_MODE or TARGET_MODE == "binary" else torch.nn.BCEWithLogitsLoss()    

    elif MODEL == "gine":
        N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
        EDGE_FEATURES = train_dataloader.dataset[0].edge_attr.shape[-1]
        if USE_FINGERPRINT and hasattr(train_dataloader.dataset[0], "fingerprint"):
            FINGERPRINT_LENGTH = len(train_dataloader.dataset[0].fingerprint)
        else:
            FINGERPRINT_LENGTH = None

        from models.GIN import GIN, GINWithEdgeFeatures
        model = GINWithEdgeFeatures(in_channels=N_FEATURES, 
                                    hidden_channels=H_DIM, 
                                    edge_dim=EDGE_FEATURES, 
                                    out_channels=len(LABELS_CODES.keys()),
                                    fingerprint_length=FINGERPRINT_LENGTH)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        loss_criterion = torch.nn.CrossEntropyLoss() 
        
    elif MODEL == "mlp":
        from models.MLP import MLP
        # N_FEATURES = the length of the extended fingerprint in the dataloader
        N_FEATURES = len(train_dataloader.dataset[0][1][0])
        # Create the model
        model = MLP(input_channels=N_FEATURES, 
                    num_categories=len(LABELS_CODES.keys())).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        loss_criterion = torch.nn.BCELoss() if "hot" in TARGET_MODE or TARGET_MODE == "binary" else torch.nn.BCEWithLogitsLoss()

    for n_run in range(N_RUNS):
        for epoch in range(1, N_EPOCHS+1):

            if MODEL == "gin":
                from models.GIN import train_epoch
                from models.GIN import evaluate 

            elif MODEL == "gine":
                from models.GIN import train_epoch
                from models.GIN import evaluate
                
            elif MODEL == "mlp":
                from models.MLP import train_epoch
                from models.MLP import evaluate

            train_avg_loss, train_precision, train_recall, train_f1, train_conf_matrix = train_epoch(model, train_dataloader, optimizer, loss_criterion, DEVICE)
            val_avg_loss, val_precision, val_recall, val_f1, val_conf_matrix = evaluate(model, val_dataloader, DEVICE, criterion=loss_criterion)
            
            if MODEL == "gin":
                GIN_TRAIN_PREC_LIST.append(train_precision)
                GIN_TRAIN_REC_LIST.append(train_recall)
                GIN_TRAIN_F1_LIST.append(train_f1)
                GIN_VAL_PREC_LIST.append(val_precision)
                GIN_VAL_REC_LIST.append(val_recall)
                GIN_VAL_F1_LIST.append(val_f1)

            elif MODEL == "gine":
                GINE_TRAIN_PREC_LIST.append(train_precision)
                GINE_TRAIN_REC_LIST.append(train_recall)
                GINE_TRAIN_F1_LIST.append(train_f1)
                GINE_VAL_PREC_LIST.append(val_precision)
                GINE_VAL_REC_LIST.append(val_recall)
                GINE_VAL_F1_LIST.append(val_f1)
                
            elif MODEL == "mlp":
                MPL_TRAIN_PREC_LIST.append(train_precision)
                MPL_TRAIN_REC_LIST.append(train_recall)
                MPL_TRAIN_F1_LIST.append(train_f1)
                MPL_VAL_PREC_LIST.append(val_precision)
                MPL_VAL_REC_LIST.append(val_recall)
                MPL_VAL_F1_LIST.append(val_f1)
                
            print(f'[TRAINING {n_run}/{N_RUNS}] Epoch: {epoch:03d}, Loss: {train_avg_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}')
            print("Training confusion matrix:")
            print(train_conf_matrix)
            
            print(f'[VALIDATION {n_run}/{N_RUNS}] Epoch: {epoch:03d}, Loss: {val_avg_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}')
            print("Validation confusion matrix:")
            print(val_conf_matrix)
            print("-"*50)

try:
    exception_trigger = sum(GIN_TRAIN_PREC_LIST)/len(GIN_TRAIN_PREC_LIST) # to trigger the exception if the list is empty
    print(f"REPORT USING GIN OVER {N_RUNS} RUNS")
    print("Avg and std of training precision, recall and f1-score:")
    print(f"Precision: {sum(GIN_TRAIN_PREC_LIST)/len(GIN_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GIN_TRAIN_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GIN_TRAIN_REC_LIST)/len(GIN_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GIN_TRAIN_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GIN_TRAIN_F1_LIST)/len(GIN_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GIN_TRAIN_F1_LIST)):.4f}")
    print("Avg and std of validation precision, recall and f1-score:")
    print(f"Precision: {sum(GIN_VAL_PREC_LIST)/len(GIN_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GIN_VAL_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GIN_VAL_REC_LIST)/len(GIN_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GIN_VAL_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GIN_VAL_F1_LIST)/len(GIN_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GIN_VAL_F1_LIST)):.4f}")
    print("-"*50)
except:
    # print("GIN model not trained")
    pass
try:
    exception_trigger = sum(GINE_TRAIN_PREC_LIST)/len(GINE_TRAIN_PREC_LIST) # to trigger the exception if the list is empty
    print(f"REPORT USING GINE OVER {N_RUNS} RUNS")
    print("Avg and std of training precision, recall and f1-score:")
    print(f"Precision: {sum(GINE_TRAIN_PREC_LIST)/len(GINE_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GINE_TRAIN_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GINE_TRAIN_REC_LIST)/len(GINE_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GINE_TRAIN_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GINE_TRAIN_F1_LIST)/len(GINE_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GINE_TRAIN_F1_LIST)):.4f}")
    print("Avg and std of validation precision, recall and f1-score:")
    print(f"Precision: {sum(GINE_VAL_PREC_LIST)/len(GINE_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GINE_VAL_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GINE_VAL_REC_LIST)/len(GINE_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GINE_VAL_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GINE_VAL_F1_LIST)/len(GINE_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GINE_VAL_F1_LIST)):.4f}")
    print("-"*50)
except:
    # print("GINE model not trained")
    pass
try:
    exception_trigger = sum(MPL_TRAIN_PREC_LIST)/len(MPL_TRAIN_PREC_LIST) # to trigger the exception if the list is empty
    print(f"REPORT USING MLP OVER {N_RUNS} RUNS")
    print("Avg and std of training precision, recall and f1-score:")
    print(f"Precision: {sum(MPL_TRAIN_PREC_LIST)/len(MPL_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(MPL_TRAIN_PREC_LIST)):.4f}")
    print(f"Recall: {sum(MPL_TRAIN_REC_LIST)/len(MPL_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(MPL_TRAIN_REC_LIST)):.4f}")
    print(f"F1-score: {sum(MPL_TRAIN_F1_LIST)/len(MPL_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(MPL_TRAIN_F1_LIST)):.4f}")
    print("Avg and std of validation precision, recall and f1-score:")
    print(f"Precision: {sum(MPL_VAL_PREC_LIST)/len(MPL_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(MPL_VAL_PREC_LIST)):.4f}")
    print(f"Recall: {sum(MPL_VAL_REC_LIST)/len(MPL_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(MPL_VAL_REC_LIST)):.4f}")
    print(f"F1-score: {sum(MPL_VAL_F1_LIST)/len(MPL_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(MPL_VAL_F1_LIST)):.4f}")
    print("-"*50)
except:
    # print("MLP model not trained")
    pass