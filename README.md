# AI Toolkit Docker Image

Docker image for training FLUX.1 and other models, based on [AI Toolkit](https://github.com/ostris/ai-toolkit).

## Quick Start
 **Shared Drive**: Mount `lora-training-shared` at `/app/workspace`

## Captioning Setup
```bash
# Navigate to /app/workspace/captions

# Make the setup script executable:
chmod +x setup_caption_env.sh

# Run the setup:
./setup_caption_env.sh
```

### After setup:
- Place images in /app/workspace/captions/input/

- Run captioning process from captions/ directory


## FLUX.1 Training
### Requirements
- GPU with **≥24GB VRAM**

⚠️ **Important**: Don't interrupt during checkpoint saves