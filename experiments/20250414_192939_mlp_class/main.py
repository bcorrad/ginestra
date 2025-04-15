import torch, os
from config import N_EPOCHS, DEVICE, MODELS, LABELS_CODES, TARGET_MODE, H_DIM, USE_FINGERPRINT, N_RUNS, TARGET_TYPE, USE_GRID_SEARCH
from alternative_dataset_builder import mlp_train_dataloader, mlp_val_dataloader, mlp_test_dataloader, gnn_train_dataloader, gnn_val_dataloader, gnn_test_dataloader

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

GAT_TRAIN_PREC_LIST = []
GAT_TRAIN_REC_LIST = []
GAT_TRAIN_F1_LIST = []
GAT_VAL_PREC_LIST = []
GAT_VAL_REC_LIST = []
GAT_VAL_F1_LIST = []

GATE_TRAIN_PREC_LIST = []
GATE_TRAIN_REC_LIST = []
GATE_TRAIN_F1_LIST = []
GATE_VAL_PREC_LIST = []
GATE_VAL_REC_LIST = []
GATE_VAL_F1_LIST = []

# Initialize a report to save the log of the training
from config import EXPERIMENT_FOLDER, N_SAMPLES, RANDOMIZE_SAMPLES, USE_AVAILABLE_DATASET, BATCH_SIZE, CLS_LIST, PATHWAYS, LABELS_CODES, MULTILABEL2MULTICLASS, USE_FINGERPRINT
import datetime
# Open the report file stream to close it at the end of the training
report_file = os.path.join(EXPERIMENT_FOLDER, "report.txt")
f = open(report_file, "w")
f.write("Training report\n")
f.write(f"Models: {MODELS}\n")
f.write(f"Target type: {TARGET_TYPE}\n")
f.write(f"Target mode: {TARGET_MODE}\n")
f.write(f"Number of epochs: {N_EPOCHS}\n")
f.write(f"Number of runs: {N_RUNS}\n")
f.write(f"Batch size: {BATCH_SIZE}\n")
f.write(f"Number of samples: {N_SAMPLES}\n")
f.write(f"Randomize samples: {RANDOMIZE_SAMPLES}\n")
f.write(f"Use available dataset: {USE_AVAILABLE_DATASET}\n")
f.write(f"Use fingerprint: {USE_FINGERPRINT}\n")
f.write(f"Use multi-label to multi-class: {MULTILABEL2MULTICLASS}\n")
f.write(f"Target classes: {CLS_LIST}\n")
f.write(f"Target pathways: {PATHWAYS}\n")
f.write(f"Experiment folder: {EXPERIMENT_FOLDER}\n")
f.write(f"Device: {DEVICE}\n")
f.write(f"Training started at: {datetime.datetime.now()}\n")
f.write("-"*50 + "\n")

for MODEL in MODELS:
    print(f"Training {MODEL.upper()} model")
    f.write(f"Training {MODEL.upper()} model\n")
    
    for n_run in range(N_RUNS):
        
        if MODEL == "gin" and gnn_train_dataloader is not None:
            train_dataloader = gnn_train_dataloader
            val_dataloader = gnn_val_dataloader
            test_dataloader = gnn_test_dataloader
            N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
            from models.GIN import GIN
            model = GIN(num_node_features=N_FEATURES, 
                        dim_h=H_DIM, 
                        num_classes=len(LABELS_CODES.keys())).to(DEVICE) #, num_heads=4
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            loss_criterion = torch.nn.CrossEntropyLoss() if "hot" in TARGET_MODE or TARGET_MODE == "binary" else torch.nn.BCEWithLogitsLoss()  
            from models.GIN import train_epoch
            from models.GIN import evaluate   

        elif MODEL == "gine" and gnn_train_dataloader is not None:
            train_dataloader = gnn_train_dataloader
            val_dataloader = gnn_val_dataloader
            test_dataloader = gnn_test_dataloader
            N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
            EDGE_FEATURES = train_dataloader.dataset[0].edge_attr.shape[-1]
            if USE_FINGERPRINT and hasattr(train_dataloader.dataset[0], "fingerprint"):
                FINGERPRINT_LENGTH = len(train_dataloader.dataset[0].fingerprint)
            else:
                FINGERPRINT_LENGTH = None

            from models.GINE import GINWithEdgeFeatures
            model = GINWithEdgeFeatures(in_channels=N_FEATURES, 
                                        hidden_channels=H_DIM, 
                                        edge_dim=EDGE_FEATURES, 
                                        out_channels=len(LABELS_CODES.keys()),
                                        fingerprint_length=FINGERPRINT_LENGTH)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            loss_criterion = torch.nn.CrossEntropyLoss() 
            from models.GINE import train_epoch
            from models.GINE import evaluate
            
        elif MODEL == "gat" and gnn_train_dataloader is not None:
            train_dataloader = gnn_train_dataloader
            val_dataloader = gnn_val_dataloader
            test_dataloader = gnn_test_dataloader
            N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
            from models.GAT import GAT
            model = GAT(in_channels=N_FEATURES, 
                        hidden_channels=H_DIM, 
                        out_channels=len(LABELS_CODES.keys())).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            loss_criterion = torch.nn.CrossEntropyLoss() 
            from models.GAT import train_epoch
            from models.GAT import evaluate
            
        elif MODEL == "gate" and gnn_train_dataloader is not None:
            train_dataloader = gnn_train_dataloader
            val_dataloader = gnn_val_dataloader
            test_dataloader = gnn_test_dataloader
            N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
            EDGE_FEATURES = train_dataloader.dataset[0].edge_attr.shape[-1]
            from models.GATE import GATE
            model = GATE(in_channels=N_FEATURES, 
                        hidden_channels=H_DIM, 
                        out_channels=len(LABELS_CODES.keys()),
                        edge_dim=EDGE_FEATURES).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            loss_criterion = torch.nn.CrossEntropyLoss() 
            from models.GATE import train_epoch
            from models.GATE import evaluate
            
        elif MODEL == "mlp" and mlp_train_dataloader is not None:
            train_dataloader = mlp_train_dataloader
            val_dataloader = mlp_val_dataloader
            test_dataloader = mlp_test_dataloader
            from models.MLP import MLP, MLP2
            from models.MLP import train_epoch
            from models.MLP import evaluate    
            if USE_GRID_SEARCH:
                # N_FEATURES = the length of the extended fingerprint in the dataloader
                N_FEATURES = len(train_dataloader.dataset[0][1][0])
                EPOCHS = 10  # Fisso, oppure includi nelle combinazioni
                import itertools
                param_grid = {
                    'units': [[4096, 2048, 1024], [6144, 3072, 1536]],
                    'dropout': [0.1, 0.2, 0.3],
                    'lr': [1e-3, 1e-4],
                    'l2_reg': [1e-2, 1e-3]}

                grid = list(itertools.product(
                        param_grid['units'],
                        param_grid['dropout'],
                        param_grid['lr'],
                        param_grid['l2_reg'])
                        )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                best_f1 = 0
                best_params = None

                for i, (units, dropout, lr, l2_reg) in enumerate(grid):
                    print(f"\n ▶️ Grid {i+1}/{len(grid)} | units={units}, dropout={dropout}, lr={lr}, l2={l2_reg}")

                    model = MLP2(input_channels=N_FEATURES,
                                 hidden1=units[0], hidden2=units[1], hidden3=units[2],
                                 dropout=dropout,
                                 num_categories=len(LABELS_CODES.keys())
                                 ).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
                    criterion = torch.nn.BCELoss()

                    for epoch in range(EPOCHS):
                        train_avg_loss, *_ = train_epoch(model, train_dataloader, optimizer, criterion, device, str(epoch+1))
                        val_avg_loss, _, _, val_f1, _ = evaluate(model, val_dataloader, device, criterion, epoch_n=str(epoch+1))
                        log_string = f'[{MODEL.upper()} TRAINING {n_run+1}/{N_RUNS}] Epoch: {epoch+1:03d}, Loss: {train_avg_loss:.4f}, Val F1-score: {val_f1:.4f}'
                        print(log_string)
                        log_string = f'[{MODEL.upper()} VALIDATION {n_run+1}/{N_RUNS}] Epoch: {epoch+1:03d}, Loss: {val_avg_loss:.4f}, Val F1-score: {val_f1:.4f}'
                        print(log_string)

                    # Puoi usare test set qui se vuoi
                    _, _, _, test_f1, _ = evaluate(model, test_dataloader, device, criterion, epoch_n="final")
                    print(f"  🧪 Test F1: {test_f1:.4f}")

                    if test_f1 > best_f1:
                        best_f1 = test_f1
                        best_params = {
                            'units': units,
                            'dropout': dropout,
                            'lr': lr,
                            'l2_reg': l2_reg
                        }

                print("\n ✅ Best configuration:", best_params)
                print(f"🏆 Best F1 score: {best_f1:.4f}")
            else:
                # Create the model
                model = MLP(input_channels=N_FEATURES, 
                                num_categories=len(LABELS_CODES.keys())).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
                loss_criterion = torch.nn.BCELoss() if "hot" in TARGET_MODE or TARGET_MODE == "binary" else torch.nn.BCEWithLogitsLoss()
                    
        
        for epoch in range(1, N_EPOCHS+1):

            train_avg_loss, train_precision, train_recall, train_f1, train_conf_matrix = train_epoch(model, train_dataloader, optimizer, loss_criterion, DEVICE, str(epoch))
            val_avg_loss, val_precision, val_recall, val_f1, val_conf_matrix = evaluate(model, val_dataloader, DEVICE, criterion=loss_criterion, epoch_n=str(epoch))
            
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
                
            elif MODEL == "gat":
                GAT_TRAIN_PREC_LIST.append(train_precision)
                GAT_TRAIN_REC_LIST.append(train_recall)
                GAT_TRAIN_F1_LIST.append(train_f1)
                GAT_VAL_PREC_LIST.append(val_precision)
                GAT_VAL_REC_LIST.append(val_recall)
                GAT_VAL_F1_LIST.append(val_f1)
                
            elif MODEL == "gate":
                GAT_TRAIN_PREC_LIST.append(train_precision)
                GAT_TRAIN_REC_LIST.append(train_recall)
                GAT_TRAIN_F1_LIST.append(train_f1)
                GAT_VAL_PREC_LIST.append(val_precision)
                GAT_VAL_REC_LIST.append(val_recall)
                GAT_VAL_F1_LIST.append(val_f1)
                
            elif MODEL == "mlp":
                MPL_TRAIN_PREC_LIST.append(train_precision)
                MPL_TRAIN_REC_LIST.append(train_recall)
                MPL_TRAIN_F1_LIST.append(train_f1)
                MPL_VAL_PREC_LIST.append(val_precision)
                MPL_VAL_REC_LIST.append(val_recall)
                MPL_VAL_F1_LIST.append(val_f1)
                
            log_string = f'[{MODEL.upper()} TRAINING {n_run+1}/{N_RUNS}] Epoch: {epoch:03d}, Loss: {train_avg_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}'
            f.write(log_string + "\n")
            print(log_string)
            if TARGET_TYPE == "pathway":
                print("Training confusion matrix:") 
                print(train_conf_matrix)
            
            log_string = f'[{MODEL.upper()} VALIDATION {n_run+1}/{N_RUNS}] Epoch: {epoch:03d}, Loss: {val_avg_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}'
            f.write(log_string + "\n")
            print(log_string)
            if TARGET_TYPE == "pathway":
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
    f.write(f"REPORT USING GIN OVER {N_RUNS} RUNS\n")
    f.write("Avg and std of training precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GIN_TRAIN_PREC_LIST)/len(GIN_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GIN_TRAIN_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GIN_TRAIN_REC_LIST)/len(GIN_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GIN_TRAIN_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GIN_TRAIN_F1_LIST)/len(GIN_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GIN_TRAIN_F1_LIST)):.4f}\n")
    f.write("Avg and std of validation precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GIN_VAL_PREC_LIST)/len(GIN_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GIN_VAL_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GIN_VAL_REC_LIST)/len(GIN_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GIN_VAL_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GIN_VAL_F1_LIST)/len(GIN_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GIN_VAL_F1_LIST)):.4f}\n")
    f.write("-"*50 + "\n")
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
    f.write(f"REPORT USING GINE OVER {N_RUNS} RUNS\n")
    f.write("Avg and std of training precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GINE_TRAIN_PREC_LIST)/len(GINE_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GINE_TRAIN_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GINE_TRAIN_REC_LIST)/len(GINE_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GINE_TRAIN_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GINE_TRAIN_F1_LIST)/len(GINE_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GINE_TRAIN_F1_LIST)):.4f}\n")
    f.write("Avg and std of validation precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GINE_VAL_PREC_LIST)/len(GINE_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GINE_VAL_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GINE_VAL_REC_LIST)/len(GINE_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GINE_VAL_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GINE_VAL_F1_LIST)/len(GINE_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GINE_VAL_F1_LIST)):.4f}\n")
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
    print("-"*50 + "\n")
    f.write(f"REPORT USING MLP OVER {N_RUNS} RUNS\n")
    f.write("Avg and std of training precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(MPL_TRAIN_PREC_LIST)/len(MPL_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(MPL_TRAIN_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(MPL_TRAIN_REC_LIST)/len(MPL_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(MPL_TRAIN_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(MPL_TRAIN_F1_LIST)/len(MPL_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(MPL_TRAIN_F1_LIST)):.4f}\n")
    f.write("Avg and std of validation precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(MPL_VAL_PREC_LIST)/len(MPL_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(MPL_VAL_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(MPL_VAL_REC_LIST)/len(MPL_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(MPL_VAL_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(MPL_VAL_F1_LIST)/len(MPL_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(MPL_VAL_F1_LIST)):.4f}\n")
    f.write("-"*50 + "\n")
except:
    # print("MLP model not trained")
    pass
try:
    exception_trigger = sum(GAT_TRAIN_PREC_LIST)/len(GAT_TRAIN_PREC_LIST) # to trigger the exception if the list is empty
    print(f"REPORT USING GAT OVER {N_RUNS} RUNS")
    print("Avg and std of training precision, recall and f1-score:")
    print(f"Precision: {sum(GAT_TRAIN_PREC_LIST)/len(GAT_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GAT_TRAIN_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GAT_TRAIN_REC_LIST)/len(GAT_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GAT_TRAIN_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GAT_TRAIN_F1_LIST)/len(GAT_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GAT_TRAIN_F1_LIST)):.4f}")
    print("Avg and std of validation precision, recall and f1-score:")
    print(f"Precision: {sum(GAT_VAL_PREC_LIST)/len(GAT_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GAT_VAL_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GAT_VAL_REC_LIST)/len(GAT_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GAT_VAL_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GAT_VAL_F1_LIST)/len(GAT_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GAT_VAL_F1_LIST)):.4f}")
    print("-"*50 + "\n")
    f.write(f"REPORT USING GAT OVER {N_RUNS} RUNS\n")
    f.write("Avg and std of training precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GAT_TRAIN_PREC_LIST)/len(GAT_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GAT_TRAIN_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GAT_TRAIN_REC_LIST)/len(GAT_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GAT_TRAIN_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GAT_TRAIN_F1_LIST)/len(GAT_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GAT_TRAIN_F1_LIST)):.4f}\n")
    f.write("Avg and std of validation precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GAT_VAL_PREC_LIST)/len(GAT_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GAT_VAL_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GAT_VAL_REC_LIST)/len(GAT_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GAT_VAL_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GAT_VAL_F1_LIST)/len(GAT_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GAT_VAL_F1_LIST)):.4f}\n")
    f.write("-"*50 + "\n")
except:
    # print("GAT model not trained")
    pass
try:
    exception_trigger = sum(GATE_TRAIN_PREC_LIST)/len(GATE_TRAIN_PREC_LIST) # to trigger the exception if the list is empty
    print(f"REPORT USING GATE OVER {N_RUNS} RUNS")
    print("Avg and std of training precision, recall and f1-score:")
    print(f"Precision: {sum(GATE_TRAIN_PREC_LIST)/len(GATE_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GATE_TRAIN_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GATE_TRAIN_REC_LIST)/len(GATE_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GATE_TRAIN_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GATE_TRAIN_F1_LIST)/len(GATE_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GATE_TRAIN_F1_LIST)):.4f}")
    print("Avg and std of validation precision, recall and f1-score:")
    print(f"Precision: {sum(GATE_VAL_PREC_LIST)/len(GATE_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GATE_VAL_PREC_LIST)):.4f}")
    print(f"Recall: {sum(GATE_VAL_REC_LIST)/len(GATE_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GATE_VAL_REC_LIST)):.4f}")
    print(f"F1-score: {sum(GATE_VAL_F1_LIST)/len(GATE_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GATE_VAL_F1_LIST)):.4f}")
    print("-"*50 + "\n")
    f.write(f"REPORT USING GATE OVER {N_RUNS} RUNS\n")
    f.write("Avg and std of training precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GATE_TRAIN_PREC_LIST)/len(GATE_TRAIN_PREC_LIST):.4f} ± {torch.std(torch.tensor(GATE_TRAIN_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GATE_TRAIN_REC_LIST)/len(GATE_TRAIN_REC_LIST):.4f} ± {torch.std(torch.tensor(GATE_TRAIN_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GATE_TRAIN_F1_LIST)/len(GATE_TRAIN_F1_LIST):.4f} ± {torch.std(torch.tensor(GATE_TRAIN_F1_LIST)):.4f}\n")
    f.write("Avg and std of validation precision, recall and f1-score:\n")
    f.write(f"Precision: {sum(GATE_VAL_PREC_LIST)/len(GATE_VAL_PREC_LIST):.4f} ± {torch.std(torch.tensor(GATE_VAL_PREC_LIST)):.4f}\n")
    f.write(f"Recall: {sum(GATE_VAL_REC_LIST)/len(GATE_VAL_REC_LIST):.4f} ± {torch.std(torch.tensor(GATE_VAL_REC_LIST)):.4f}\n")
    f.write(f"F1-score: {sum(GATE_VAL_F1_LIST)/len(GATE_VAL_F1_LIST):.4f} ± {torch.std(torch.tensor(GATE_VAL_F1_LIST)):.4f}\n")
    f.write("-"*50 + "\n")
except:
    # print("GATE model not trained")
    pass
