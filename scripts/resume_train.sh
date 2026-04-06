#!/bin/bash
# =============================================================================
# GhostLM Auto-Resume Training Script
# =============================================================================
#
# Automatically resumes training from the latest checkpoint if the process
# crashes or is interrupted. Retries up to MAX_RETRIES times with a configurable
# delay between attempts. All output is logged with timestamps.
#
# Usage:
#   chmod +x scripts/resume_train.sh
#   ./scripts/resume_train.sh
#
# Customize the variables below to adjust training parameters.
# =============================================================================

# --- Configuration ---
PRESET="ghost-tiny"
MAX_STEPS=10000
BATCH_SIZE=2
DEVICE="cpu"
GRAD_ACCUM=4
CHECKPOINT_DIR="checkpoints"
LOG_FILE="logs/resume_train.log"
MAX_RETRIES=5
RETRY_DELAY=30

# --- Setup ---
mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# --- Banner ---
echo "╔══════════════════════════════════════════════╗"
echo "║     GhostLM Auto-Resume Training             ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Preset:    $PRESET                          "
echo "║  Max Steps: $MAX_STEPS                       "
echo "║  Batch:     $BATCH_SIZE                      "
echo "║  Device:    $DEVICE                          "
echo "║  Grad Acc:  $GRAD_ACCUM                      "
echo "║  Max Retry: $MAX_RETRIES                     "
echo "╚══════════════════════════════════════════════╝"
echo ""

# --- Retry Loop ---
attempt=1

while [ $attempt -le $MAX_RETRIES ]; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Attempt $attempt / $MAX_RETRIES — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Determine if we should resume from checkpoint
    CHECKPOINT_FLAG=""
    if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
        CHECKPOINT_FLAG="--checkpoint $CHECKPOINT_DIR/best_model.pt"
        echo "  Found checkpoint: best_model.pt — resuming..."
    else
        echo "  No checkpoint found — starting fresh training..."
    fi

    # Build and run the training command
    CMD="python scripts/train.py \
        --preset $PRESET \
        --max-steps $MAX_STEPS \
        --batch-size $BATCH_SIZE \
        --device $DEVICE \
        --grad-accum $GRAD_ACCUM \
        $CHECKPOINT_FLAG"

    echo ""
    echo "  Running: $CMD"
    echo ""

    # Execute and log with timestamps
    $CMD 2>&1 | while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done | tee -a "$LOG_FILE"

    # Check exit code from the training command
    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Training completed successfully!"
        exit 0
    else
        echo ""
        echo "❌ Training crashed with exit code $EXIT_CODE (attempt $attempt / $MAX_RETRIES)."

        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "   Retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        else
            echo ""
            echo "⛔ Max retries reached ($MAX_RETRIES). Exiting."
            echo "   Check $LOG_FILE for details."
            exit 1
        fi
    fi

    attempt=$((attempt + 1))
done
