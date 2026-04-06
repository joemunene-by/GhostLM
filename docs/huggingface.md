# Uploading GhostLM to HuggingFace Hub

## Prerequisites
- HuggingFace account at huggingface.co
- A trained checkpoint at checkpoints/best_model.pt
- huggingface_hub installed: pip install huggingface_hub

## Step 1: Create HuggingFace Account
1. Go to huggingface.co and sign up
2. Go to Settings → Access Tokens
3. Create a new token with Write access
4. Copy the token

## Step 2: Create the Model Repository
1. Go to huggingface.co/new
2. Name it: GhostLM-tiny (or GhostLM-small when ready)
3. Set to Public
4. Click Create

## Step 3: Upload via Script
```bash
python scripts/push_to_hub.py \
  --checkpoint checkpoints/best_model.pt \
  --repo-id YOUR_USERNAME/GhostLM-tiny \
  --token YOUR_HF_TOKEN
```

## Step 4: Verify Upload
Visit https://huggingface.co/YOUR_USERNAME/GhostLM-tiny
You should see:
- pytorch_model.pt
- config.json
- tokenizer_config.json
- README.md (from MODEL_CARD.md)

## Step 5: Deploy to HuggingFace Spaces (Free Demo)
1. Go to huggingface.co/new-space
2. Name: GhostLM-demo
3. SDK: Gradio
4. Upload demo/app.py as app.py
5. Add requirements: gradio, torch, tiktoken
6. The Space will auto-build and deploy

## Naming Convention
- joemunene/GhostLM-tiny — ghost-tiny weights
- joemunene/GhostLM-small — ghost-small weights (coming soon)
- joemunene/GhostLM — organization page

## After Upload
Update README.md Training Progress table with:
- HuggingFace model link
- Download instructions using huggingface_hub
