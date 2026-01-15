import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree

# =========================
# FILES
# =========================
DATA_FILE = r"C:\addmm\NN_results\data.csv"
EXP_FILE  = r"C:\addmm\NN_results\isihara_gt_step100_mesh_disp.csv"
OUT_MODEL = "best_surrogate.pt"


# =========================
# OUTPUT FOLDER
# =========================
OUT_DIR = r"C:\addmm\NN_results\results_nn_task2"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================
SEED = 0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

BATCH_SIZE = 1024
EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE = 50

WIDTH = 128
DEPTH = 5
ACTIVATION = "tanh"

INPUT_COLS = ["x", "y", "C10", "C01", "C20", "invD"]
TARGET_COLS = ["ux", "uy"]

# =========================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_activation(name):
    if name == "tanh": return nn.Tanh()
    if name == "relu": return nn.ReLU()

# =========================
class TabDataset(Dataset):
    def __init__(self, df, X_mean, X_std, y_mean, y_std):
        X = df[INPUT_COLS].values.astype(np.float32)
        y = df[TARGET_COLS].values.astype(np.float32)
        self.X = (X - X_mean) / X_std
        self.y = (y - y_mean) / y_std
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.float32)

# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, activation):
        super().__init__()
        layers = [nn.Linear(in_dim, width), get_activation(activation)]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), get_activation(activation)]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# =========================
set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(DATA_FILE)

# Use first 20 groups
groups_all = np.sort(df["group_id"].unique())
pick_groups = groups_all[:20]
df = df[df["group_id"].isin(pick_groups)]

groups = df["group_id"].unique()
np.random.shuffle(groups)

n = len(groups)
n_train = int(TRAIN_RATIO*n)
n_val   = int(VAL_RATIO*n)

train_groups = set(groups[:n_train])
val_groups   = set(groups[n_train:n_train+n_val])
test_groups  = set(groups[n_train+n_val:])

train_df = df[df.group_id.isin(train_groups)]
val_df   = df[df.group_id.isin(val_groups)]
test_df  = df[df.group_id.isin(test_groups)]

# Normalization
X_train = train_df[INPUT_COLS].values.astype(np.float32)
y_train = train_df[TARGET_COLS].values.astype(np.float32)

X_mean, X_std = X_train.mean(0), X_train.std(0)+1e-8
y_mean, y_std = y_train.mean(0), y_train.std(0)+1e-8

train_loader = DataLoader(TabDataset(train_df,X_mean,X_std,y_mean,y_std),batch_size=1024,shuffle=True)
val_loader   = DataLoader(TabDataset(val_df,X_mean,X_std,y_mean,y_std),batch_size=1024)
test_loader  = DataLoader(TabDataset(test_df,X_mean,X_std,y_mean,y_std),batch_size=1024)

model = MLP(6,2,WIDTH,DEPTH,ACTIVATION).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

best_val = 1e30
bad = 0

# =========================
# TRAINING
# =========================
for epoch in range(EPOCHS):
    model.train()
    for Xb,yb in train_loader:
        Xb,yb = Xb.to(device), yb.to(device)
        loss = loss_fn(model(Xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb,yb in val_loader:
            Xb,yb = Xb.to(device), yb.to(device)
            val_loss += loss_fn(model(Xb), yb).item()*len(Xb)
    val_loss /= len(val_df)

    if epoch % 25 == 0:
        print(f"Epoch {epoch:4d} | Val Loss = {val_loss:.4e}")

    if val_loss < best_val:
        best_val = val_loss
        bad = 0
        torch.save(model.state_dict(), OUT_MODEL)
    else:
        bad += 1
        if bad > PATIENCE:
            print("Early stopping")
            break

model.load_state_dict(torch.load(OUT_MODEL))
model.eval()

# =========================
# PICK ONE TEST GROUP
# =========================
pick_gid = list(test_groups)[0]
sub = df[df.group_id==pick_gid]

X = sub[INPUT_COLS].values.astype(np.float32)
y_true = sub[TARGET_COLS].values.astype(np.float32)

Xn = (X-X_mean)/X_std
with torch.no_grad():
    y_pred = model(torch.tensor(Xn,dtype=torch.float32).to(device)).cpu().numpy()*y_std+y_mean

# =========================
# TRUE vs PRED
# =========================
plt.scatter(y_true[:,0],y_pred[:,0],s=6)
plt.title("ux true vs pred")
plt.savefig(f"{OUT_DIR}/01_true_vs_pred_ux.png",dpi=200)
plt.show()

plt.scatter(y_true[:,1],y_pred[:,1],s=6)
plt.title("uy true vs pred")
plt.savefig(f"{OUT_DIR}/02_true_vs_pred_uy.png",dpi=200)
plt.show()

# =========================
# PREDICTED FIELDS
# =========================
plt.scatter(sub.x,sub.y,c=y_pred[:,0],s=8,cmap="jet")
plt.colorbar(); plt.title("Predicted ux")
plt.savefig(f"{OUT_DIR}/03_predicted_ux.png",dpi=200)
plt.show()

plt.scatter(sub.x,sub.y,c=y_pred[:,1],s=8,cmap="jet")
plt.colorbar(); plt.title("Predicted uy")
plt.savefig(f"{OUT_DIR}/04_predicted_uy.png",dpi=200)
plt.show()

# =========================
# EXPERIMENTAL
# =========================
exp = pd.read_csv(EXP_FILE)
exp = exp[exp["type"]=="node"]

exp_x = exp.x.values.astype(np.float32)
exp_y = exp.y.values.astype(np.float32)
exp_ux = exp.ux.values.astype(np.float32)
exp_uy = exp.uy.values.astype(np.float32)

C10,C01,C20,invD = sub.iloc[0][["C10","C01","C20","invD"]]

Xexp = np.column_stack([exp_x,exp_y,
                        np.full_like(exp_x,C10),
                        np.full_like(exp_x,C01),
                        np.full_like(exp_x,C20),
                        np.full_like(exp_x,invD)]).astype(np.float32)

Xexp_n = (Xexp-X_mean)/X_std
with torch.no_grad():
    yexp_pred = model(torch.tensor(Xexp_n,dtype=torch.float32).to(device)).cpu().numpy()*y_std+y_mean

plt.scatter(exp_ux,yexp_pred[:,0],s=6)
plt.title("Experimental vs Pred ux")
plt.savefig(f"{OUT_DIR}/05_exp_vs_pred_ux.png",dpi=200)
plt.show()

plt.scatter(exp_uy,yexp_pred[:,1],s=6)
plt.title("Experimental vs Pred uy")
plt.savefig(f"{OUT_DIR}/06_exp_vs_pred_uy.png",dpi=200)
plt.show()

# =========================
# DEFORMED + ERROR
# =========================
x_nn = sub.x.values
y_nn = sub.y.values
ux_nn = y_pred[:,0]
uy_nn = y_pred[:,1]

x_nn_def = x_nn + ux_nn
y_nn_def = y_nn + uy_nn

exp_x_def = exp_x + exp_ux
exp_y_def = exp_y + exp_uy

tree = cKDTree(np.column_stack([exp_x_def, exp_y_def]))
_, idx = tree.query(np.column_stack([x_nn_def, y_nn_def]))

err_ux = ux_nn - exp_ux[idx]
err_uy = uy_nn - exp_uy[idx]

# Deformed NN
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(x_nn_def,y_nn_def,c=ux_nn,s=8,cmap="jet")
plt.title("NN ux (deformed)")
plt.axis("equal"); plt.colorbar()

plt.subplot(1,2,2)
plt.scatter(x_nn_def,y_nn_def,c=uy_nn,s=8,cmap="jet")
plt.title("NN uy (deformed)")
plt.axis("equal"); plt.colorbar()
plt.savefig(f"{OUT_DIR}/07_nn_deformed.png",dpi=200)
plt.show()

# Error
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(x_nn_def,y_nn_def,c=err_ux,s=10,cmap="viridis")
plt.title("Error ux (NN − EXP)")
plt.axis("equal"); plt.colorbar()

plt.subplot(1,2,2)
plt.scatter(x_nn_def,y_nn_def,c=err_uy,s=10,cmap="viridis")
plt.title("Error uy (NN − EXP)")
plt.axis("equal"); plt.colorbar()
plt.savefig(f"{OUT_DIR}/08_error_deformed.png",dpi=200)
plt.show()

print("All figures saved in:", OUT_DIR)



#