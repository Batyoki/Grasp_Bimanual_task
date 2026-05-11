#!/bin/bash
################################################################################
# PARALLEL TRAINING MONITOR & AGGREGATOR
# ════════════════════════════════════════════════════════════════════════════
# Real-time monitoring of parallel training instances
# Video aggregation and results collection
#
# Usage:
#   bash monitor_parallel.sh                    # Monitor active jobs
#   bash monitor_parallel.sh --collect          # Aggregate results
#   bash monitor_parallel.sh --watch            # Real-time watch mode
################################################################################

set -e

BASE_DIR="${HOME}/yash"
TRAINRL_DIR="${BASE_DIR}/trainRL"
LOGS_DIR="${TRAINRL_DIR}/logs"
PARALLEL_LOGS_DIR="${LOGS_DIR}/parallel_runs"
AGGREGATE_DIR="${LOGS_DIR}/videos_aggregate"

COMMAND=${1:-"status"}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  PARALLEL TRAINING MONITOR                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

################################################################################
# Function: Show Status
################################################################################
show_status() {
    echo "📊 PARALLEL INSTANCES STATUS"
    echo "═" * 60
    
    if [[ ! -d "${PARALLEL_LOGS_DIR}" ]]; then
        echo "❌ No parallel training directory found"
        return
    fi
    
    # Find all instance directories
    instance_count=0
    for instance_dir in "${PARALLEL_LOGS_DIR}"/instance_*/; do
        if [[ -d "${instance_dir}" ]]; then
            instance_id=$(basename "${instance_dir}")
            instance_count=$((instance_count + 1))
            
            echo ""
            echo "Instance: $instance_id"
            
            # Check if training is running
            if [[ -f "${instance_dir}/training.log" ]]; then
                log_file="${instance_dir}/training.log"
                
                # Get last few lines
                tail_lines=$(tail -3 "${log_file}")
                echo "  Latest output:"
                echo "$tail_lines" | sed 's/^/    /'
                
                # Count steps
                steps=$(grep -o "Step [0-9]*" "${log_file}" | tail -1 | awk '{print $2}')
                if [[ -n "${steps}" ]]; then
                    echo "  Steps completed: ${steps}"
                fi
                
                # Count videos
                video_count=$(ls "${instance_dir}/videos"/*.mp4 2>/dev/null | wc -l || echo 0)
                echo "  Videos recorded: ${video_count}"
                
                # Count checkpoints
                ckpt_count=$(ls "${instance_dir}/checkpoints"/*.pt 2>/dev/null | wc -l || echo 0)
                echo "  Checkpoints saved: ${ckpt_count}"
            else
                echo "  Status: Not started"
            fi
        fi
    done
    
    echo ""
    echo "Total instances: $instance_count"
    echo ""
}

################################################################################
# Function: Collect Results
################################################################################
collect_results() {
    echo "📦 COLLECTING RESULTS"
    echo "═" * 60
    
    if [[ ! -d "${PARALLEL_LOGS_DIR}" ]]; then
        echo "❌ No parallel training directory found"
        return
    fi
    
    total_videos=0
    total_plots=0
    total_checkpoints=0
    
    # Create aggregate directories if they don't exist
    mkdir -p "${LOGS_DIR}/videos_aggregate"
    mkdir -p "${LOGS_DIR}/plots_aggregate"
    mkdir -p "${LOGS_DIR}/checkpoints_aggregate"
    
    for instance_dir in "${PARALLEL_LOGS_DIR}"/instance_*/; do
        if [[ -d "${instance_dir}" ]]; then
            instance_name=$(basename "${instance_dir}")
            
            # Copy videos
            video_dir="${instance_dir}/videos"
            if [[ -d "${video_dir}" ]] && [[ -n "$(ls "${video_dir}"/*.mp4 2>/dev/null)" ]]; then
                cp "${video_dir}"/*.mp4 "${LOGS_DIR}/videos_aggregate/" 2>/dev/null || true
                video_count=$(ls "${video_dir}"/*.mp4 2>/dev/null | wc -l)
                total_videos=$((total_videos + video_count))
                echo "✓ Copied ${video_count} videos from $instance_name"
            fi
            
            # Copy plots
            plot_dir="${instance_dir}/plots"
            if [[ -d "${plot_dir}" ]] && [[ -n "$(ls "${plot_dir}"/*.png 2>/dev/null)" ]]; then
                cp "${plot_dir}"/*.png "${LOGS_DIR}/plots_aggregate/" 2>/dev/null || true
                plot_count=$(ls "${plot_dir}"/*.png 2>/dev/null | wc -l)
                total_plots=$((total_plots + plot_count))
                echo "✓ Copied ${plot_count} plots from $instance_name"
            fi
            
            # Copy best checkpoint
            ckpt="${instance_dir}/checkpoints/checkpoint_best.pt"
            if [[ -f "${ckpt}" ]]; then
                cp "${ckpt}" "${LOGS_DIR}/checkpoints_aggregate/checkpoint_${instance_name}_best.pt"
                total_checkpoints=$((total_checkpoints + 1))
                echo "✓ Copied best checkpoint from $instance_name"
            fi
        fi
    done
    
    echo ""
    echo "Summary:"
    echo "  Total videos: $total_videos"
    echo "  Total plots: $total_plots"
    echo "  Total checkpoints: $total_checkpoints"
    echo "  Aggregate dir: ${LOGS_DIR}/videos_aggregate/"
    echo ""
}

################################################################################
# Function: Watch Mode
################################################################################
watch_mode() {
    echo "👁️  WATCH MODE (Ctrl+C to exit)"
    echo "═" * 60
    echo ""
    
    while true; do
        clear
        echo "📊 PARALLEL TRAINING MONITOR [$(date)]"
        echo "═" * 60
        echo ""
        
        show_status
        
        # Check SLURM job array status if available
        if command -v squeue &> /dev/null; then
            echo ""
            echo "SLURM Job Array Status:"
            squeue -a --name=bimanual_rl_parallel -o "%.8i %.20j %.8T %.10M %.6D %R" || echo "(No active jobs)"
        fi
        
        sleep 10
    done
}

################################################################################
# Function: Generate Report
################################################################################
generate_report() {
    echo "📄 GENERATING TRAINING REPORT"
    echo "═" * 60
    echo ""
    
    report_file="${LOGS_DIR}/PARALLEL_TRAINING_REPORT_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║        PARALLEL TRAINING COMPLETION REPORT                    ║"
        echo "║        Generated: $(date)                        ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo ""
        
        for instance_dir in "${PARALLEL_LOGS_DIR}"/instance_*/; do
            if [[ -d "${instance_dir}" ]]; then
                instance_name=$(basename "${instance_dir}")
                echo ""
                echo "─ $instance_name ─────────────────────────────────────────"
                
                if [[ -f "${instance_dir}/training.log" ]]; then
                    log_file="${instance_dir}/training.log"
                    
                    # Extract key statistics
                    echo "Training Log: $log_file"
                    
                    final_line=$(tail -1 "${log_file}")
                    echo "Final Status: $final_line"
                    
                    video_count=$(ls "${instance_dir}/videos"/*.mp4 2>/dev/null | wc -l || echo 0)
                    echo "Videos: $video_count"
                    
                    ckpt_count=$(ls "${instance_dir}/checkpoints"/*.pt 2>/dev/null | wc -l || echo 0)
                    echo "Checkpoints: $ckpt_count"
                    
                    if [[ -f "${instance_dir}/checkpoints/checkpoint_best.pt.json" ]]; then
                        echo "Best Checkpoint Metadata:"
                        cat "${instance_dir}/checkpoints/checkpoint_best.pt.json" | sed 's/^/  /'
                    fi
                fi
            fi
        done
        
        echo ""
        echo "═" * 60
        echo "Aggregated Results:"
        echo "  Videos: $(ls ${LOGS_DIR}/videos_aggregate/*.mp4 2>/dev/null | wc -l || echo 0)"
        echo "  Plots: $(ls ${LOGS_DIR}/plots_aggregate/*.png 2>/dev/null | wc -l || echo 0)"
        echo "  Checkpoints: $(ls ${LOGS_DIR}/checkpoints_aggregate/*.pt 2>/dev/null | wc -l || echo 0)"
        
    } | tee "${report_file}"
    
    echo ""
    echo "Report saved to: $report_file"
}

################################################################################
# Function: Cleanup Old Results
################################################################################
cleanup_old() {
    echo "🧹 CLEANING UP OLD RESULTS"
    echo "═" * 60
    
    if [[ ! -d "${PARALLEL_LOGS_DIR}" ]]; then
        echo "No parallel training directory to clean"
        return
    fi
    
    # Remove old instance directories (keep last 5)
    instance_count=$(find "${PARALLEL_LOGS_DIR}" -maxdepth 1 -type d -name "instance_*" | wc -l)
    
    if [[ $instance_count -gt 5 ]]; then
        echo "Found $instance_count instances, keeping last 5..."
        find "${PARALLEL_LOGS_DIR}" -maxdepth 1 -type d -name "instance_*" -printf '%T@ %p\n' | \
            sort -n | head -n $((instance_count - 5)) | cut -d' ' -f2- | \
            while read old_dir; do
                echo "Removing: $old_dir"
                rm -rf "$old_dir"
            done
    fi
    
    echo "✓ Cleanup complete"
}

################################################################################
# Main
################################################################################

case "${COMMAND}" in
    status|--status)
        show_status
        ;;
    collect|--collect)
        collect_results
        ;;
    report|--report)
        generate_report
        ;;
    watch|--watch)
        watch_mode
        ;;
    cleanup|--cleanup)
        cleanup_old
        ;;
    *)
        echo "Usage: $0 {status|collect|report|watch|cleanup}"
        echo ""
        echo "Commands:"
        echo "  status    - Show current status of all instances"
        echo "  collect   - Aggregate videos, plots, checkpoints"
        echo "  report    - Generate training completion report"
        echo "  watch     - Real-time monitoring (Ctrl+C to exit)"
        echo "  cleanup   - Remove old instance directories"
        echo ""
        show_status
        ;;
esac
