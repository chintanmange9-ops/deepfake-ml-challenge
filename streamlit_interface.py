import streamlit as st
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, json, hashlib, re, tempfile, zipfile
import torch.nn.functional as F
from torch import nn
from web3 import Web3

# ==========================================
# HELPERS
# ==========================================
def alphanum_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', s)]

def sha256_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

class TestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.images = [os.path.join(folder, f) for f in os.listdir(folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.images.sort(key=lambda x: alphanum_key(os.path.basename(x)))
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, path

def connect_blockchain(url, abi_json, address):
    try:
        web3 = Web3(Web3.HTTPProvider(url))
        if not web3.is_connected():
            return None, None, None, "❌ Failed to connect"
        abi = json.loads(abi_json)
        contract = web3.eth.contract(address=address, abi=abi)
        accounts = web3.eth.accounts
        return web3, contract, accounts, "✅ Connected to Ganache"
    except Exception as e:
        return None, None, None, f"❌ Error: {e}"

# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Deepfake Detection + Blockchain", layout="wide")
st.title("🧠 Deepfake Detection + Blockchain Storage (DeepfakeLedger)")

# --- STEP 1: UPLOAD MODEL ---
st.header("Step 1: Upload Trained Model (.pth)")
model_file = st.file_uploader("Upload model file", type=["pth"])
if not model_file:
    st.warning("Please upload your model (.pth) file.")
    st.stop()

temp_model_path = os.path.join(tempfile.gettempdir(), "uploaded_model.pth")
with open(temp_model_path, "wb") as f:
    f.write(model_file.read())

# --- STEP 2: UPLOAD TEST IMAGES ZIP ---
st.header("Step 2: Upload ZIP file containing test images")
zip_file = st.file_uploader("Upload ZIP of test images", type=["zip"])
if not zip_file:
    st.warning("Please upload a ZIP file containing images.")
    st.stop()

temp_dir = tempfile.mkdtemp()
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)
st.success(f"✅ Extracted images to temporary folder: {temp_dir}")

# --- STEP 3: CHOOSE BLOCKCHAIN OPTION ---
st.header("Step 3: Blockchain Storage Option")
use_bc = st.radio("Do you want to upload results to Blockchain?", ["No", "Yes"])
web3, contract, account = None, None, None
ganache_url, contract_address, abi_data = None, None, None

if use_bc == "Yes":
    st.subheader("Blockchain Configuration")
    ganache_url = st.text_input("Ganache URL", "HTTP://127.0.0.1:7545")
    contract_address = st.text_input("Contract Address")
    abi_file = st.file_uploader("Upload ABI JSON file", type=["json"])

    if abi_file:
        abi_data = abi_file.read().decode("utf-8")
        web3, contract, accounts, msg = connect_blockchain(ganache_url, abi_data, contract_address)
        st.info(msg)
        if web3:
            account = st.selectbox("Select Ganache Account", accounts)
else:
    st.info("Results will only be saved locally as JSON.")

# --- STEP 4: LOAD MODEL ---
st.header("Step 4: Load Model and Configure")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

try:
    # Load state dict
    state_dict = torch.load(temp_model_path, map_location=device)

    # Detect head structure
    has_seq_head = any(k.startswith("fc.1") or k.startswith("fc.4") for k in state_dict.keys())
    if has_seq_head:
        st.info("Detected custom sequential classifier head.")
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    else:
        st.info("Detected standard single-layer classifier head.")
        base_model.fc = nn.Linear(base_model.fc.in_features, 2)

    base_model.load_state_dict(state_dict, strict=False)
    base_model.to(device).eval()
    TEMPERATURE = 1.8
    st.success(f"✅ Model loaded successfully on {device.type.upper()} (T={TEMPERATURE})")

except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

model = base_model

# --- STEP 5: RUN INFERENCE ---
st.header("Step 5: Run Deepfake Inference")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = TestDataset(temp_dir, transform)
if len(dataset) == 0:
    st.error("No valid images found in the uploaded ZIP.")
    st.stop()

loader = DataLoader(dataset, batch_size=8, shuffle=False)
if st.button("🚀 Run Inference"):
    preds, progress = [], st.progress(0)
    logs = st.empty()

    with torch.no_grad():
        for i, (imgs, paths) in enumerate(loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits / 1.8, dim=1)
            confs, labels = torch.max(probs, 1)
            for path, lbl, conf in zip(paths, labels, confs):
                label = "real" if lbl.item() == 0 else "fake"
                preds.append({
                    "image": os.path.basename(path),
                    "prediction": label,
                    "confidence": round(float(conf.item()), 4)
                })
            progress.progress((i + 1) / len(loader))
            logs.text(f"Processed {i+1}/{len(loader)} batches")

    st.success("✅ Inference Complete!")
    st.dataframe(preds[:10])

    # thumbnails
    cols = st.columns(min(5, len(preds)))
    for i, col in enumerate(cols):
        if i < len(preds):
            img_path = os.path.join(temp_dir, preds[i]["image"])
            col.image(img_path, caption=f"{preds[i]['prediction']} ({preds[i]['confidence']})", width="stretch")

    # --- STEP 6: BLOCKCHAIN UPLOAD OR LOCAL SAVE ---
    st.header("Step 6: Save or Upload Results")
    failed_uploads = []
    if use_bc == "Yes" and web3 and contract and account:
        for item in preds:
            image_path = os.path.join(temp_dir, item["image"])
            img_hash = sha256_hash(image_path)
            confidence = int(item["confidence"] * 100)
            try:
                tx = contract.functions.storeRecord(img_hash, item["prediction"], confidence).transact({"from": account})
                web3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                failed_uploads.append({"image": item["image"], "error": str(e)})

        if failed_uploads:
            st.error(f"❌ {len(failed_uploads)} uploads failed.")
            st.json(failed_uploads)
        else:
            st.success("🎉 All predictions uploaded successfully to blockchain.")

    # Always allow JSON export
    out_json = os.path.join(tempfile.gettempdir(), "predictions.json")
    with open(out_json, "w") as f:
        json.dump(preds, f, indent=4)
    st.download_button(
        label="⬇️ Download Results as JSON",
        data=json.dumps(preds, indent=4),
        file_name="deepfake_predictions.json",
        mime="application/json"
    )
