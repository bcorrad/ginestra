import torch

from config import N_EPOCHS as num_epochs, DEVICE as device, MODEL, LABELS_CODES, TARGET_MODE, H_DIM, USE_FINGERPRINT
# metrics_average_mode = "two_classes" if TARGET_MODE == "two_classes" else "micro"
from dataset_builder_correct import train_dataloader, val_dataloader
BCE_THRESHOLD = 0.5
if MODEL.lower() == "gin":
    N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
    from models.GIN_model import GIN, GINWithEdgeFeatures
    model = GIN(num_node_features=N_FEATURES, dim_h=H_DIM, num_classes=len(LABELS_CODES.keys())).to(device) #, num_heads=4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_criterion = torch.nn.CrossEntropyLoss() if "hot" in TARGET_MODE or TARGET_MODE == "binary" else torch.nn.BCEWithLogitsLoss()    

elif MODEL.lower() == "gine":
    N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
    EDGE_FEATURES = train_dataloader.dataset[0].edge_attr.shape[-1]
    if USE_FINGERPRINT and hasattr(train_dataloader.dataset[0], "fingerprint"):
        FINGERPRINT_LENGTH = len(train_dataloader.dataset[0].fingerprint)
    else:
        FINGERPRINT_LENGTH = None

    from models.GIN_model import GIN, GINWithEdgeFeatures
    model = GINWithEdgeFeatures(in_channels=N_FEATURES, 
                                hidden_channels=H_DIM, 
                                edge_dim=EDGE_FEATURES, 
                                out_channels=len(LABELS_CODES.keys()),
                                fingerprint_length=FINGERPRINT_LENGTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_criterion = torch.nn.CrossEntropyLoss() 

elif MODEL.lower() == "new_gine":
    from models.new_GINE import GINE
    N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
    EDGE_FEATURES = train_dataloader.dataset[0].edge_attr.shape[-1]
    if USE_FINGERPRINT and hasattr(train_dataloader.dataset[0], "fingerprint"):
        FINGERPRINT_LENGTH = len(train_dataloader.dataset[0].fingerprint)
    else:
        FINGERPRINT_LENGTH = None

    model = GINE(num_node_features=N_FEATURES, 
                 edge_dim=EDGE_FEATURES, 
                 dim_h=H_DIM, 
                 num_classes=len(LABELS_CODES.keys()), 
                 fingerprint_dim=FINGERPRINT_LENGTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_criterion = torch.nn.CrossEntropyLoss() 

elif MODEL.lower() == "gat":
    N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
    EDGE_FEATURES = train_dataloader.dataset[0].edge_attr.shape[-1]
    from models.GIN_model import GATWithEdgeFeatures as GAT
    model = GAT(in_channels=N_FEATURES, hidden_channels=H_DIM, out_channels=len(LABELS_CODES.keys()), edge_dim=EDGE_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_criterion = torch.nn.CrossEntropyLoss() if "hot" in TARGET_MODE or TARGET_MODE == "binary" else torch.nn.BCEWithLogitsLoss()

elif MODEL.lower() == "gae":
    N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
    from models.GAE_model import GCNEncoder, GAE, GINEncoder
    # Inizializzazione dell'encoder e del modello GAE
    encoder = GINEncoder(in_channels=N_FEATURES, hidden_channels=H_DIM, out_channels=len(LABELS_CODES.keys()))
    model = GAE(encoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

elif MODEL.lower() == "mlp":
    N_FEATURES = train_dataloader.dataset.tensors[0].shape[-1]
    from models.MLP_model import Model
    model = Model(input_channels=N_FEATURES, num_categories=len(LABELS_CODES.keys())).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    loss_criterion = torch.nn.BCELoss()  # Binary cross-entropy loss

elif MODEL.lower() == "gat_gine":
    N_FEATURES = train_dataloader.dataset[0].x.shape[-1]
    EDGE_FEATURES = train_dataloader.dataset[0].edge_attr.shape[-1]
    from models.model_test import HybridGAT_GINE
    model = HybridGAT_GINE(in_channels=N_FEATURES, hidden_channels=H_DIM, edge_dim=EDGE_FEATURES, out_channels=len(LABELS_CODES.keys())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_criterion = torch.nn.CrossEntropyLoss() if "hot" in TARGET_MODE or TARGET_MODE == "binary" else torch.nn.BCEWithLogitsLoss()


from utils.earlystop import EarlyStopper
early_stopper = EarlyStopper(patience=10, min_delta=1e-5)

for epoch in range(1, num_epochs+1):

    if MODEL.lower() == "gin":
        from models.GIN_model import train_epoch
        from models.GIN_model import evaluate 

        # train_avg_loss, train_precision, train_recall, train_f1, train_conf_matrix = train_epoch(model, train_dataloader, optimizer, loss_criterion, device)
        # val_precision, val_recall, val_f1, val_conf_matrix = evaluate(model, val_dataloader, device)
        
        # print(f'[TRAINING] Epoch: {epoch:03d}, Loss: {train_avg_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}')
        # print("Training confusion matrix:")
        # print(train_conf_matrix)
        
        # print(f'[VALIDATION] Epoch: {epoch:03d}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}')
        # print("Validation confusion matrix:")
        # print(val_conf_matrix)
        
        # print("-"*50)

    elif MODEL.lower() == "gine":

        from models.GIN_model import train_epoch
        from models.GIN_model import evaluate
        
        # train_avg_loss, train_precision, train_recall, train_f1, train_conf_matrix = train_epoch(model, train_dataloader, optimizer, loss_criterion, device)
        # val_precision, val_recall, val_f1, val_conf_matrix = evaluate(model, val_dataloader, device)
        
        # print(f'[TRAINING] Epoch: {epoch:03d}, Loss: {train_avg_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}')
        # print("Training confusion matrix:")
        # print(train_conf_matrix)
        
        # print(f'[VALIDATION] Epoch: {epoch:03d}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}')
        # print("Validation confusion matrix:")
        # print(val_conf_matrix)
        
        # print("-"*50)
    
    elif MODEL.lower() == "new_gine":

        from models.GIN_model import train_epoch
        from models.GIN_model import evaluate
    
    elif MODEL.lower() == "gae":
        from models.GAE_model import train_epoch
        from models.GAE_model import evaluate
        # GAE_train_loss = train_epoch(model, optimizer, train_dataloader)
        # GAE_val_loss = evaluate(model, val_dataloader)
        
        # print(f'[TRAINING] Epoch: {epoch:03d}, Loss: {GAE_train_loss:.4f}')
        # print(f'[VALIDATION] Epoch: {epoch:03d}, Loss: {GAE_val_loss:.4f}')
        # print("-"*50)

    if MODEL.lower() == "mlp":
        from models.MLP_model import train_epoch
        from models.MLP_model import evaluate
        
        # train_avg_loss, train_precision, train_recall, train_f1, train_conf_matrix = train_epoch_mlp(model, train_dataloader, optimizer, loss_fn, device)
        # val_precision, val_recall, val_f1, val_conf_matrix = evaluate_mlp(model, val_dataloader, device)

        # print(f'[TRAINING] Epoch: {epoch:03d}, Loss: {train_avg_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}')
        # print("Training confusion matrix:")
        # print(train_conf_matrix)

        # print(f'[VALIDATION] Epoch: {epoch:03d}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}')
        # print("Validation confusion matrix:")
        # print(val_conf_matrix)

        # print("-"*50)

    if MODEL.lower() == "gat":
        from models.GIN_model import train_epoch
        from models.GIN_model import evaluate
        
        # train_avg_loss, train_precision, train_recall, train_f1, train_conf_matrix = train_epoch(model, train_dataloader, optimizer, loss_criterion, device)
        # val_precision, val_recall, val_f1, val_conf_matrix = evaluate(model, val_dataloader, device)
        
        # print(f'[TRAINING] Epoch: {epoch:03d}, Loss: {train_avg_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}')
        # print("Training confusion matrix:")
        # print(train_conf_matrix)
        
        # print(f'[VALIDATION] Epoch: {epoch:03d}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}')
        # print("Validation confusion matrix:")
        # print(val_conf_matrix)
        
        # print("-"*50)

    train_avg_loss, train_precision, train_recall, train_f1, train_conf_matrix = train_epoch(model, train_dataloader, optimizer, loss_criterion, device)
    val_avg_loss, val_precision, val_recall, val_f1, val_conf_matrix = evaluate(model, val_dataloader, device, criterion=loss_criterion)

    # if early_stopper.early_stop(val_avg_loss, train_avg_loss, model.state_dict()):
    #     print(f"Early stopping at epoch {epoch}")
    #     break
    
    print(f'[TRAINING] Epoch: {epoch:03d}, Loss: {train_avg_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}, Training F1-score: {train_f1:.4f}')
    print("Training confusion matrix:")
    print(train_conf_matrix)
    
    print(f'[VALIDATION] Epoch: {epoch:03d}, Loss: {val_avg_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-score: {val_f1:.4f}')
    print("Validation confusion matrix:")
    print(val_conf_matrix)
    
    print("-"*50)




